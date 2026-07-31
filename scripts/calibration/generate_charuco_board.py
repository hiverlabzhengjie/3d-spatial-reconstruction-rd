"""Generate the canonical S01 A4 ChArUco calibration target as a PDF."""

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

SQUARES_X: Final = 6
SQUARES_Y: Final = 8
SQUARE_LENGTH_MM: Final = 30.0
MARKER_LENGTH_MM: Final = 22.0
BOARD_WIDTH_MM: Final = SQUARES_X * SQUARE_LENGTH_MM
BOARD_HEIGHT_MM: Final = SQUARES_Y * SQUARE_LENGTH_MM
PIXELS_PER_MM: Final = 20

DICTIONARY_NAME: Final = "DICT_5X5_100"
DICTIONARY_ID: Final = cv2.aruco.DICT_5X5_100
BOARD_ID: Final = "s01-charuco-6x8-30mm-5x5-100-v1"


def _mm(value: float) -> float:
    return value * POINTS_PER_MM


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_object(number: int, body: bytes) -> bytes:
    return f"{number} 0 obj\n".encode() + body + b"\nendobj\n"


def _stream_object(number: int, dictionary: str, payload: bytes) -> bytes:
    header = f"<< {dictionary} /Length {len(payload)} >>\nstream\n".encode()
    return _pdf_object(number, header + payload + b"\nendstream")


def _create_board_image() -> NDArray[np.uint8]:
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH_MM / 1000.0,
        MARKER_LENGTH_MM / 1000.0,
        dictionary,
    )
    width_px = int(BOARD_WIDTH_MM * PIXELS_PER_MM)
    height_px = int(BOARD_HEIGHT_MM * PIXELS_PER_MM)
    image = board.generateImage(
        (width_px, height_px),
        marginSize=0,
        borderBits=1,
    )
    if image.shape != (height_px, width_px):
        raise RuntimeError(f"Unexpected board image shape: {image.shape}")
    if image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected board image dtype: {image.dtype}")
    return cast(NDArray[np.uint8], image)


def _page_content() -> bytes:
    board_x_mm = (A4_WIDTH_MM - BOARD_WIDTH_MM) / 2.0
    board_y_mm = (A4_HEIGHT_MM - BOARD_HEIGHT_MM) / 2.0
    board_x = _mm(board_x_mm)
    board_y = _mm(board_y_mm)
    board_width = _mm(BOARD_WIDTH_MM)
    board_height = _mm(BOARD_HEIGHT_MM)

    title_y = _mm(284.0)
    subtitle_y = _mm(277.5)
    ruler_y = _mm(13.0)
    ruler_start_x = _mm(55.0)
    ruler_length = _mm(100.0)
    subtitle = (
        f"{BOARD_ID} | {SQUARES_X}x{SQUARES_Y} squares | "
        f"square {SQUARE_LENGTH_MM:.0f} mm | "
        f"marker {MARKER_LENGTH_MM:.0f} mm | {DICTIONARY_NAME}"
    )

    commands = [
        "q",
        f"{board_width:.6f} 0 0 {board_height:.6f} {board_x:.6f} {board_y:.6f} cm",
        "/BoardImage Do",
        "Q",
        "0 g",
        "BT",
        "/Helvetica-Bold 9 Tf",
        f"{_mm(15.0):.6f} {title_y:.6f} Td",
        f"({_pdf_escape('S01 Canonical ChArUco Board')}) Tj",
        "ET",
        "BT",
        "/Helvetica 6.5 Tf",
        f"{_mm(15.0):.6f} {subtitle_y:.6f} Td",
        f"({_pdf_escape(subtitle)}) Tj",
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
            f"{_mm(80.5):.6f} {_mm(7.0):.6f} Td",
            "(100 mm print-scale check) Tj",
            "ET",
            "BT",
            "/Helvetica 5.5 Tf",
            f"{_mm(15.0):.6f} {_mm(3.5):.6f} Td",
            (
                "(Print A4 portrait at 100% / Actual Size. "
                "Disable Fit, Shrink, and Scale to Page.) Tj"
            ),
            "ET",
        ]
    )
    return ("\n".join(commands) + "\n").encode("ascii")


def _build_pdf(image: NDArray[np.uint8]) -> bytes:
    page_width = _mm(A4_WIDTH_MM)
    page_height = _mm(A4_HEIGHT_MM)
    height_px, width_px = image.shape
    compressed_image = zlib.compress(image.tobytes(), level=9)
    content = _page_content()

    objects = [
        _pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        _pdf_object(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        _pdf_object(
            3,
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {page_width:.6f} {page_height:.6f}] "
                "/Resources << /XObject << /BoardImage 5 0 R >> "
                "/Font << /Helvetica 6 0 R /Helvetica-Bold 7 0 R >> >> "
                "/Contents 4 0 R >>"
            ).encode(),
        ),
        _stream_object(4, "", content),
        _stream_object(
            5,
            (
                f"/Type /XObject /Subtype /Image /Width {width_px} "
                f"/Height {height_px} /ColorSpace /DeviceGray "
                "/BitsPerComponent 8 /Filter /FlateDecode"
            ),
            compressed_image,
        ),
        _pdf_object(
            6,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ),
        _pdf_object(
            7,
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ),
    ]

    header = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    body = bytearray(header)
    offsets = [0]
    for obj in objects:
        offsets.append(len(body))
        body.extend(obj)

    xref_offset = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    return bytes(body)


def _write_manifest(output_path: Path, pdf_bytes: bytes) -> None:
    manifest = {
        "board_id": BOARD_ID,
        "dictionary": DICTIONARY_NAME,
        "squares_x": SQUARES_X,
        "squares_y": SQUARES_Y,
        "square_length_mm": SQUARE_LENGTH_MM,
        "marker_length_mm": MARKER_LENGTH_MM,
        "board_width_mm": BOARD_WIDTH_MM,
        "board_height_mm": BOARD_HEIGHT_MM,
        "paper": "A4 portrait",
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
        default=Path("output/pdf/s01_charuco_6x8_30mm_a4.pdf"),
    )
    args = parser.parse_args()

    image = _create_board_image()
    pdf_bytes = _build_pdf(image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(pdf_bytes)
    _write_manifest(args.output, pdf_bytes)
    print(args.output)


if __name__ == "__main__":
    main()
