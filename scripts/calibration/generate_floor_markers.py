"""Generate the canonical S01 A4 floor-marker set as a multi-page PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
from pathlib import Path
from typing import Final, cast

import cv2
import numpy as np
from numpy.typing import NDArray

A4_WIDTH_MM: Final = 210.0
A4_HEIGHT_MM: Final = 297.0
POINTS_PER_MM: Final = 72.0 / 25.4

MARKER_IDS: Final = (40, 41, 42, 43)
MARKER_LENGTH_MM: Final = 180.0
PIXELS_PER_MM: Final = 20
DICTIONARY_NAME: Final = "DICT_5X5_100"
DICTIONARY_ID: Final = cv2.aruco.DICT_5X5_100
MARKER_SET_ID: Final = "s01-floor-markers-40-43-180mm-5x5-100-v1"


def _mm(value: float) -> float:
    return value * POINTS_PER_MM


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode() + body + b"\nendobj\n"


def _stream_object(number: int, dictionary: str, payload: bytes) -> bytes:
    header = f"<< {dictionary} /Length {len(payload)} >>\nstream\n".encode()
    return _pdf_object(number, header + payload + b"\nendstream")


def _create_marker_image(marker_id: int) -> NDArray[np.uint8]:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    side_pixels = int(MARKER_LENGTH_MM * PIXELS_PER_MM)
    image = cv2.aruco.generateImageMarker(
        dictionary,
        marker_id,
        side_pixels,
        borderBits=1,
    )
    if image.shape != (side_pixels, side_pixels):
        raise RuntimeError(f"Unexpected marker image shape: {image.shape}")
    if image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected marker image dtype: {image.dtype}")
    return cast(NDArray[np.uint8], image)


def _page_content(marker_id: int, image_name: str) -> bytes:
    marker_x_mm = (A4_WIDTH_MM - MARKER_LENGTH_MM) / 2.0
    marker_y_mm = 55.0
    marker_x = _mm(marker_x_mm)
    marker_y = _mm(marker_y_mm)
    marker_size = _mm(MARKER_LENGTH_MM)

    role = "WORLD ORIGIN (0, 0, 0)" if marker_id == 40 else "SURVEYED FLOOR MARKER"
    title = f"M{marker_id} - {role}"
    subtitle = (
        f"{MARKER_SET_ID} | marker ID {marker_id} | "
        f"{MARKER_LENGTH_MM:.0f} mm | {DICTIONARY_NAME}"
    )

    ruler_y = _mm(23.0)
    ruler_start_x = _mm(55.0)
    ruler_length = _mm(100.0)
    commands = [
        "q",
        f"{marker_size:.6f} 0 0 {marker_size:.6f} "
        f"{marker_x:.6f} {marker_y:.6f} cm",
        f"/{image_name} Do",
        "Q",
        "0 g",
        "BT",
        "/Helvetica-Bold 11 Tf",
        f"{_mm(15.0):.6f} {_mm(285.0):.6f} Td",
        f"({_pdf_escape(title)}) Tj",
        "ET",
        "BT",
        "/Helvetica 6.5 Tf",
        f"{_mm(15.0):.6f} {_mm(277.5):.6f} Td",
        f"({_pdf_escape(subtitle)}) Tj",
        "ET",
        "BT",
        "/Helvetica-Bold 8 Tf",
        f"{_mm(65.0):.6f} {_mm(247.0):.6f} Td",
        "(PAGE TOP = +Y) Tj",
        "ET",
        "BT",
        "/Helvetica-Bold 8 Tf",
        f"{_mm(67.0):.6f} {_mm(47.0):.6f} Td",
        "(PAGE RIGHT = +X) Tj",
        "ET",
        "0.35 w",
        f"{ruler_start_x:.6f} {ruler_y:.6f} m",
        f"{ruler_start_x + ruler_length:.6f} {ruler_y:.6f} l",
        "S",
    ]

    for index in range(11):
        x = ruler_start_x + _mm(index * 10.0)
        tick_height = _mm(3.0 if index in {0, 5, 10} else 2.0)
        commands.extend(
            [
                f"{x:.6f} {ruler_y - tick_height / 2.0:.6f} m",
                f"{x:.6f} {ruler_y + tick_height / 2.0:.6f} l",
                "S",
            ]
        )

    commands.extend(
        [
            "BT",
            "/Helvetica 6.5 Tf",
            f"{_mm(80.5):.6f} {_mm(17.0):.6f} Td",
            "(100 mm print-scale check) Tj",
            "ET",
            "BT",
            "/Helvetica 5.5 Tf",
            f"{_mm(15.0):.6f} {_mm(7.0):.6f} Td",
            (
                "(Print A4 portrait at 100% / Actual Size. "
                "Disable Fit, Shrink, and Scale to Page.) Tj"
            ),
            "ET",
        ]
    )
    return ("\n".join(commands) + "\n").encode("ascii")


def _build_pdf(images: dict[int, NDArray[np.uint8]]) -> bytes:
    page_width = _mm(A4_WIDTH_MM)
    page_height = _mm(A4_HEIGHT_MM)
    catalog_number = 1
    pages_number = 2
    regular_font_number = 3
    bold_font_number = 4

    object_bodies: dict[int, bytes] = {}
    page_numbers: list[int] = []
    next_number = 5
    for marker_id in MARKER_IDS:
        page_number = next_number
        content_number = next_number + 1
        image_number = next_number + 2
        next_number += 3
        page_numbers.append(page_number)

        image_name = f"Marker{marker_id}"
        content = _page_content(marker_id, image_name)
        image = images[marker_id]
        height_px, width_px = image.shape
        compressed_image = zlib.compress(image.tobytes(), level=9)

        object_bodies[page_number] = (
            f"<< /Type /Page /Parent {pages_number} 0 R "
            f"/MediaBox [0 0 {page_width:.6f} {page_height:.6f}] "
            f"/Resources << /XObject << /{image_name} {image_number} 0 R >> "
            f"/Font << /Helvetica {regular_font_number} 0 R "
            f"/Helvetica-Bold {bold_font_number} 0 R >> >> "
            f"/Contents {content_number} 0 R >>"
        ).encode()
        object_bodies[content_number] = (
            f"<< /Length {len(content)} >>\nstream\n".encode()
            + content
            + b"\nendstream"
        )
        object_bodies[image_number] = (
            (
                f"<< /Type /XObject /Subtype /Image /Width {width_px} "
                f"/Height {height_px} /ColorSpace /DeviceGray "
                f"/BitsPerComponent 8 /Filter /FlateDecode "
                f"/Length {len(compressed_image)} >>\nstream\n"
            ).encode()
            + compressed_image
            + b"\nendstream"
        )

    kids = " ".join(f"{number} 0 R" for number in page_numbers)
    object_bodies[catalog_number] = (
        f"<< /Type /Catalog /Pages {pages_number} 0 R >>".encode()
    )
    object_bodies[pages_number] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {len(page_numbers)} >>".encode()
    )
    object_bodies[regular_font_number] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )
    object_bodies[bold_font_number] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"
    )

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray(header)
    offsets = [0]
    for number in range(1, next_number):
        offsets.append(len(body))
        body.extend(_pdf_object(number, object_bodies[number]))

    xref_offset = len(body)
    body.extend(f"xref\n0 {next_number}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(
        (
            f"trailer\n<< /Size {next_number} /Root {catalog_number} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(body)


def _write_manifest(output_path: Path, pdf_bytes: bytes) -> None:
    manifest = {
        "marker_set_id": MARKER_SET_ID,
        "dictionary": DICTIONARY_NAME,
        "marker_ids": list(MARKER_IDS),
        "marker_length_mm": MARKER_LENGTH_MM,
        "paper": "A4 portrait",
        "page_top_direction": "+Y",
        "page_right_direction": "+X",
        "print_scale_percent": 100,
        "pdf_sha256": hashlib.sha256(pdf_bytes).hexdigest(),
    }
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/pdf/s01_floor_markers_40_43_180mm_a4.pdf"),
    )
    args = parser.parse_args()

    images = {marker_id: _create_marker_image(marker_id) for marker_id in MARKER_IDS}
    pdf_bytes = _build_pdf(images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf_bytes)
    _write_manifest(args.output, pdf_bytes)
    print(args.output)


if __name__ == "__main__":
    main()
