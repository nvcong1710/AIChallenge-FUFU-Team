"""CLI: python -m app.ingest.cli <file|dir>...

Tự động phát hiện media type (.mp4 → video, .mp3 → audio, .jpg → image)
và dispatch sang pipeline tương ứng.
"""

import argparse

from ..common.types import MediaType
from .pipeline import run_ingest
from .utils import collect_files, group_by_type


def main():
    parser = argparse.ArgumentParser(
        description="Ingest multimedia (video/audio/image) vào FUFU v2."
    )
    parser.add_argument("paths", nargs="+", help="File hoặc thư mục")
    parser.add_argument(
        "--only",
        choices=["video", "audio", "image"],
        help="Chỉ ingest 1 media type",
    )
    args = parser.parse_args()

    files = collect_files(args.paths)
    if not files:
        print("Không có file nào hợp lệ.")
        return

    groups = group_by_type(files)
    print(
        f"Phát hiện: {len(groups[MediaType.VIDEO])} video, "
        f"{len(groups[MediaType.AUDIO])} audio, "
        f"{len(groups[MediaType.IMAGE])} image"
    )

    if args.only:
        mt = MediaType(args.only)
        files = groups[mt]
        print(f"--only {args.only}: ingest {len(files)} file.")

    run_ingest(files)


if __name__ == "__main__":
    main()
