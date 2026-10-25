import hashlib
import io
import math
import os
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image

APP_TITLE = "Dotted Circle Counter"
APP_VERSION = "Streamlit V12"
FILE_KEYWORD = "no sew"
SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")

WORK_SIZE = 2500
MIN_DIAMETER = 8
MAX_DIAMETER_RATIO = 0.06
MIN_CIRCULARITY = 0.75
MIN_FILL_RATIO = 0.70
ASPECT_MIN, ASPECT_MAX = 0.8, 1.25
DOT_CENTER_TOL = 0.35
DOT_MAX_AREA_RATIO = 0.35
HOLLOW_MIN_HOLE_RATIO = 0.10
HOLLOW_CENTER_TOL = 0.20
ISOLATION_MARGIN = 1.0
MIN_SHAPE_AREA_RATIO = 0.005
VERTEX_ZONE = 0.22
EDGE_ZONE = 0.12
MAX_PREVIEW_WIDTH = 1800

st.set_page_config(page_title=APP_TITLE, page_icon="⭕", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1800px;}
    [data-testid="stMetric"] {background:#f6f8fb;border:1px solid #e3e8ef;border-radius:12px;padding:14px;}
    [data-testid="stFileUploaderDropzone"] {min-height:150px;border-radius:14px;}
    .app-note {padding:10px 14px;border-radius:10px;background:#eef6ff;border:1px solid #cfe6ff;}
    @media (prefers-color-scheme: dark) {
      [data-testid="stMetric"] {background:#172033;border-color:#34425c;}
      .app-note {background:#13263a;border-color:#234466;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state():
    defaults = {
        "records": [],
        "upload_signature": "",
        "processing_signature": "",
        "selected_name": None,
        "last_error": "",
        "cancel_requested": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


def read_image_bytes(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Không đọc được nội dung ảnh")
    return image


def encode_image(image, extension=".png"):
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("Không thể mã hóa ảnh kết quả")
    return buffer.tobytes()


def children(hierarchy, index):
    result = []
    child = hierarchy[index][2]
    while child >= 0:
        result.append(child)
        child = hierarchy[child][0]
    return result


def centroid(contour):
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def find_shapes(contours, shape):
    height, width = shape
    minimum_area = MIN_SHAPE_AREA_RATIO * height * width
    shapes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if 3 <= len(polygon) <= 12:
            shapes.append((polygon.reshape(-1, 2).astype(np.float32), area))
    return shapes


def is_isolated(binary_without_frame, x, y, diameter):
    height, width = binary_without_frame.shape
    margin = int(ISOLATION_MARGIN * diameter)
    half = int(diameter / 2) + 3
    x1 = max(0, int(x) - half - margin)
    x2 = min(width, int(x) + half + margin)
    y1 = max(0, int(y) - half - margin)
    y2 = min(height, int(y) + half + margin)
    window = binary_without_frame[y1:y2, x1:x2].copy()
    if window.size == 0:
        return False
    cv2.circle(window, (int(x) - x1, int(y) - y1), half + 1, 0, -1)
    return np.count_nonzero(window) < 0.01 * window.size


def near_vertex(x, y, radius, shapes):
    for polygon, area in shapes:
        size = math.sqrt(area)
        if size <= 0 or 2 * radius > 0.25 * size:
            continue
        contour = polygon.reshape(-1, 1, 2)
        edge_distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        if edge_distance <= 0 or edge_distance > EDGE_ZONE * size:
            continue
        vertex_distance = min(math.hypot(x - px, y - py) for px, py in polygon)
        if vertex_distance <= VERTEX_ZONE * size:
            return True
    return False


def contour_depth(hierarchy, index):
    depth = 0
    parent = hierarchy[index][3]
    while parent >= 0:
        depth += 1
        parent = hierarchy[parent][3]
    return depth


def deduplicate(items):
    result = []
    for item in sorted(items, key=lambda value: -value[2]):
        if all(
            math.hypot(item[0] - existing[0], item[1] - existing[1])
            > max(item[2], existing[2])
            for existing in result
        ):
            result.append(item)
    return result


def detect_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = gray.shape
    scale = 1.0

    if max(original_height, original_width) < WORK_SIZE:
        scale = WORK_SIZE / float(max(original_height, original_width))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    found = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    maximum_diameter = MAX_DIAMETER_RATIO * max(binary.shape)
    shapes = find_shapes(contours, binary.shape)

    line_width = max(3, int(0.004 * max(binary.shape)))
    frame_mask = np.zeros_like(binary)
    minimum_frame_area = MIN_SHAPE_AREA_RATIO * binary.size
    for contour in contours:
        if cv2.contourArea(contour) >= minimum_frame_area:
            cv2.drawContours(frame_mask, [contour], -1, 255, line_width)
    binary_without_frame = cv2.bitwise_and(binary, cv2.bitwise_not(frame_mask))

    dotted = []
    hollow = []

    for index, contour in enumerate(contours):
        if contour_depth(hierarchy, index) % 2 == 1:
            continue

        child_items = children(hierarchy, index)
        if len(child_items) != 1:
            continue

        bx, by, width, height = cv2.boundingRect(contour)
        if width < MIN_DIAMETER or height < MIN_DIAMETER:
            continue
        if width > maximum_diameter or height > maximum_diameter:
            continue
        if not ASPECT_MIN <= width / float(height) <= ASPECT_MAX:
            continue

        x = bx + width / 2.0
        y = by + height / 2.0
        radius = (width + height) / 4.0
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < MIN_CIRCULARITY:
            continue
        if area / (math.pi * radius * radius) < MIN_FILL_RATIO:
            continue

        hole_index = child_items[0]
        hole_area = cv2.contourArea(contours[hole_index])
        hole_center = centroid(contours[hole_index])
        if hole_center is None:
            continue
        if math.hypot(hole_center[0] - x, hole_center[1] - y) > HOLLOW_CENTER_TOL * radius:
            continue

        inner_items = children(hierarchy, hole_index)
        if len(inner_items) == 1 and hole_area > 0.25 * area:
            dot_contour = contours[inner_items[0]]
            dot_center = centroid(dot_contour)
            if (
                dot_center
                and math.hypot(dot_center[0] - x, dot_center[1] - y) < DOT_CENTER_TOL * radius
                and cv2.contourArea(dot_contour) < DOT_MAX_AREA_RATIO * hole_area
            ):
                dotted.append((x, y, radius))
        elif not inner_items and hole_area >= HOLLOW_MIN_HOLE_RATIO * area:
            if is_isolated(binary_without_frame, x, y, 2 * radius) and near_vertex(x, y, radius, shapes):
                hollow.append((x, y, radius))

    dotted = deduplicate(dotted)
    hollow = [
        item
        for item in deduplicate(hollow)
        if all(
            math.hypot(item[0] - dot[0], item[1] - dot[1])
            > max(item[2], dot[2]) * 1.5
            for dot in dotted
        )
    ]

    results = [(x, y, radius, "dot") for x, y, radius in dotted]
    results += [(x, y, radius, "hollow") for x, y, radius in hollow]
    results = [
        (
            int(round(x / scale)),
            int(round(y / scale)),
            max(2, int(round(radius / scale))),
            kind,
        )
        for x, y, radius, kind in results
    ]
    return sorted(results, key=lambda value: (value[1], value[0]))


def draw_result(image, circles):
    output = image.copy()
    height, width = output.shape[:2]
    thickness = max(2, int(round(max(height, width) / 1500)))
    font_scale = max(0.55, min(1.2, max(height, width) / 1800))

    for number, (x, y, radius, kind) in enumerate(circles, start=1):
        color = (0, 200, 0) if kind == "dot" else (0, 140, 255)
        cv2.circle(output, (x, y), radius + 6, color, thickness, cv2.LINE_AA)
        cv2.putText(
            output,
            str(number),
            (x + radius + 8, max(20, y - radius - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            thickness,
            cv2.LINE_AA,
        )

    top_bar = max(55, int(height * 0.045))
    output = cv2.copyMakeBorder(output, top_bar, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    dotted_count = sum(1 for item in circles if item[3] == "dot")
    hollow_count = sum(1 for item in circles if item[3] == "hollow")
    label = f"TOTAL: {len(circles)} | DOTTED: {dotted_count} | HOLLOW: {hollow_count}"
    cv2.putText(output, label, (12, int(top_bar * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (0, 0, 255), 2, cv2.LINE_AA)
    return output


def safe_filename(name):
    return Path(name.replace("\\", "/")).name


def uploaded_relative_name(uploaded):
    name = str(getattr(uploaded, "name", "image"))
    return name.replace("\\", "/")


def valid_uploaded_files(uploaded_files):
    valid = []
    duplicates = set()
    for uploaded in uploaded_files or []:
        relative_name = uploaded_relative_name(uploaded)
        base_name = safe_filename(relative_name)
        extension = Path(base_name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            continue
        if FILE_KEYWORD not in base_name.lower():
            continue
        data = uploaded.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        key = (relative_name.lower(), digest)
        if key in duplicates:
            continue
        duplicates.add(key)
        valid.append({"name": base_name, "relative_name": relative_name, "data": data, "digest": digest})
    valid.sort(key=lambda item: item["relative_name"].lower())
    return valid


def upload_signature(files):
    text = "|".join(f"{item['relative_name']}:{item['digest']}" for item in files)
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else ""


def make_table(records):
    rows = []
    for index, record in enumerate(records, start=1):
        rows.append(
            {
                "STT": index,
                "Tên bản vẽ": record["relative_name"],
                "Vòng có chấm": record["dotted_count"] if record["ok"] else None,
                "Vòng rỗng": record["hollow_count"] if record["ok"] else None,
                "Tổng số lượng": record["count"] if record["ok"] else None,
                "Trạng thái": record["status"],
            }
        )
    return pd.DataFrame(rows)


def make_excel(records):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Ket qua"
    headers = ["STT", "Tên bản vẽ", "Vòng có chấm", "Vòng rỗng", "Tổng số lượng", "Trạng thái"]
    sheet.append(headers)

    thin = Side(style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border

    for index, record in enumerate(records, start=1):
        sheet.append(
            [
                index,
                record["relative_name"],
                record["dotted_count"] if record["ok"] else None,
                record["hollow_count"] if record["ok"] else None,
                record["count"] if record["ok"] else None,
                record["status"],
            ]
        )
        if not record["ok"]:
            sheet.cell(sheet.max_row, 6).font = Font(color="C50F1F")

    first_data_row = 2
    last_data_row = max(2, sheet.max_row)
    sheet.append(
        [
            "",
            "TỔNG",
            f"=SUM(C{first_data_row}:C{last_data_row})",
            f"=SUM(D{first_data_row}:D{last_data_row})",
            f"=SUM(E{first_data_row}:E{last_data_row})",
            "",
        ]
    )
    for cell in sheet[sheet.max_row]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="DDEBF7")

    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        for cell in row:
            cell.border = border
            cell.alignment = Alignment(vertical="center")
        row[0].alignment = Alignment(horizontal="center")
        for position in (2, 3, 4):
            row[position].alignment = Alignment(horizontal="center")

    widths = {"A": 8, "B": 58, "C": 16, "D": 14, "E": 16, "F": 28}
    for column, width in widths.items():
        sheet.column_dimensions[column].width = width
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = f"A1:F{last_data_row}"

    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def make_result_zip(records):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        used_names = set()
        for index, record in enumerate(records, start=1):
            if not record["ok"]:
                continue
            stem = Path(record["name"]).stem
            candidate = f"{stem}_result.png"
            if candidate.lower() in used_names:
                candidate = f"{index:03d}_{candidate}"
            used_names.add(candidate.lower())
            archive.writestr(candidate, record["result_bytes"])
    return output.getvalue()


def process_files(files):
    records = []
    progress = st.progress(0, text="Đang chuẩn bị...")
    status = st.empty()
    total = len(files)

    for index, item in enumerate(files, start=1):
        status.info(f"Đang xử lý {index}/{total}: {item['relative_name']}")
        record = {
            "name": item["name"],
            "relative_name": item["relative_name"],
            "digest": item["digest"],
            "count": 0,
            "dotted_count": 0,
            "hollow_count": 0,
            "circles": [],
            "status": "",
            "ok": False,
            "original_bytes": item["data"],
            "result_bytes": b"",
        }
        try:
            image = read_image_bytes(item["data"])
            circles = detect_circles(image)
            result_image = draw_result(image, circles)
            record["circles"] = circles
            record["count"] = len(circles)
            record["dotted_count"] = sum(1 for value in circles if value[3] == "dot")
            record["hollow_count"] = sum(1 for value in circles if value[3] == "hollow")
            record["result_bytes"] = encode_image(result_image)
            record["status"] = "Hoàn tất"
            record["ok"] = True
        except Exception as exc:
            record["status"] = f"Lỗi: {exc}"
        records.append(record)
        progress.progress(index / total, text=f"Đã xử lý {index}/{total}")

    status.success(f"Hoàn tất {total} bản vẽ.")
    return records


st.title("⭕ Đếm vòng tròn trên bản vẽ NO SEW")
st.caption(f"{APP_VERSION} · Kéo thả folder hoặc nhiều ảnh · Lọc tự động tên chứa NO SEW")

with st.sidebar:
    st.subheader("Thiết lập")
    st.write("Định dạng hỗ trợ:")
    st.code(", ".join(SUPPORTED_EXTENSIONS), language=None)
    st.info("Ứng dụng chỉ phân tích ảnh có tên chứa NO SEW, không phân biệt chữ hoa và chữ thường.")
    if st.button("Xóa toàn bộ kết quả", use_container_width=True):
        st.session_state.records = []
        st.session_state.upload_signature = ""
        st.session_state.processing_signature = ""
        st.session_state.selected_name = None
        st.rerun()

st.markdown(
    '<div class="app-note"><b>Cách dùng:</b> kéo cả folder vào vùng bên dưới. Nếu trình duyệt không nhận folder, bấm Browse files và chọn nhiều ảnh cùng lúc.</div>',
    unsafe_allow_html=True,
)

try:
    uploaded_files = st.file_uploader(
        "Kéo thả folder hoặc các file ảnh vào đây",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
        accept_multiple_files="directory",
        help="Giữ nguyên cấu trúc tên tương đối khi trình duyệt hỗ trợ chọn folder.",
    )
except TypeError:
    uploaded_files = st.file_uploader(
        "Kéo thả nhiều file ảnh vào đây",
        type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

files = valid_uploaded_files(uploaded_files)
current_signature = upload_signature(files)
all_uploaded_count = len(uploaded_files or [])
filtered_out = max(0, all_uploaded_count - len(files))

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("File đã tải", all_uploaded_count)
metric2.metric("Bản vẽ NO SEW", len(files))
metric3.metric("Đã xử lý", len(st.session_state.records))
metric4.metric("Tổng ký hiệu", sum(record["count"] for record in st.session_state.records if record["ok"]))

if filtered_out:
    st.warning(f"Đã bỏ qua {filtered_out} file do tên không chứa NO SEW hoặc định dạng không được hỗ trợ.")

new_upload = bool(current_signature and current_signature != st.session_state.processing_signature)
button_label = "Phân tích bản vẽ" if not st.session_state.records else "Phân tích lại folder"
run = st.button(button_label, type="primary", disabled=not files, use_container_width=True)

if run:
    st.session_state.records = process_files(files)
    st.session_state.upload_signature = current_signature
    st.session_state.processing_signature = current_signature
    st.session_state.selected_name = st.session_state.records[0]["relative_name"] if st.session_state.records else None
    st.rerun()

if new_upload and st.session_state.records:
    st.info("Folder/file đã thay đổi. Bấm **Phân tích lại folder** để cập nhật kết quả.")

records = st.session_state.records
if not records:
    if files:
        st.info(f"Đã sẵn sàng phân tích {len(files)} bản vẽ.")
    else:
        st.info("Chưa có bản vẽ hợp lệ. Hãy kéo folder hoặc tải các ảnh có tên chứa NO SEW.")
    st.stop()

st.divider()
left, right = st.columns([1.08, 1.55], gap="large")

with left:
    st.subheader("Kết quả")
    table = make_table(records)
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=min(620, 78 + len(table) * 35),
        column_config={
            "STT": st.column_config.NumberColumn(width="small", format="%d"),
            "Tên bản vẽ": st.column_config.TextColumn(width="large"),
            "Vòng có chấm": st.column_config.NumberColumn(width="small", format="%d"),
            "Vòng rỗng": st.column_config.NumberColumn(width="small", format="%d"),
            "Tổng số lượng": st.column_config.NumberColumn(width="small", format="%d"),
            "Trạng thái": st.column_config.TextColumn(width="medium"),
        },
    )

    excel_data = make_excel(records)
    zip_data = make_result_zip(records)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download1, download2 = st.columns(2)
    with download1:
        st.download_button(
            "Tải Excel",
            data=excel_data,
            file_name=f"NoSew_Circle_Count_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with download2:
        st.download_button(
            "Tải ảnh kết quả ZIP",
            data=zip_data,
            file_name=f"NoSew_Result_Images_{timestamp}.zip",
            mime="application/zip",
            use_container_width=True,
        )

with right:
    st.subheader("Xem bản vẽ")
    options = [record["relative_name"] for record in records]
    default_index = 0
    if st.session_state.selected_name in options:
        default_index = options.index(st.session_state.selected_name)
    selected = st.selectbox("Chọn bản vẽ", options, index=default_index)
    st.session_state.selected_name = selected
    record = next(item for item in records if item["relative_name"] == selected)

    if record["ok"]:
        summary1, summary2, summary3 = st.columns(3)
        summary1.metric("Tổng", record["count"])
        summary2.metric("Có chấm", record["dotted_count"])
        summary3.metric("Vòng rỗng", record["hollow_count"])
        st.image(record["result_bytes"], caption=record["relative_name"], use_container_width=True)
        st.download_button(
            "Tải ảnh đang xem",
            data=record["result_bytes"],
            file_name=f"{Path(record['name']).stem}_result.png",
            mime="image/png",
            use_container_width=True,
        )
        st.caption("Màu xanh: vòng có chấm · Màu cam: vòng rỗng · Số đỏ: thứ tự nhận diện")
    else:
        st.error(record["status"])
        st.image(record["original_bytes"], caption=record["relative_name"], use_container_width=True)
