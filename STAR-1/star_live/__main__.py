"""
python -m star_live                          # kamera 0 + 1, önizleme açık
python -m star_live --cameras 0 2            # kamera 0 ve 2
python -m star_live --host 192.168.1.10      # uzak Blender
python -m star_live --no-preview             # GUI olmadan
"""
from __future__ import annotations

import argparse

from .stream import StarLiveStream


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m star_live",
        description="STAR-1 – 2 kamera canlı iskelet akışı → Blender 5.0",
    )
    parser.add_argument(
        "--cameras", nargs="+", type=int, default=[0, 1],
        metavar="ID", help="Kamera indeksleri (varsayılan: 0 1)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Blender addon UDP host (varsayılan: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=7777,
        help="Blender addon UDP port (varsayılan: 7777)",
    )
    parser.add_argument(
        "--width", type=int, default=640,
        help="Kamera çözünürlüğü genişliği (varsayılan: 640)",
    )
    parser.add_argument(
        "--height", type=int, default=480,
        help="Kamera çözünürlüğü yüksekliği (varsayılan: 480)",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="İstenen FPS (varsayılan: 30)",
    )
    parser.add_argument(
        "--no-preview", dest="preview", action="store_false", default=True,
        help="Kamera önizlemesini kapat",
    )
    parser.add_argument(
        "--smooth-cutoff", type=float, default=1.5, metavar="HZ",
        help="One Euro min cutoff frekansı — düşük=daha fazla smooth (varsayılan: 1.5)",
    )
    parser.add_argument(
        "--smooth-beta", type=float, default=0.3, metavar="B",
        help="One Euro hız katsayısı — yüksek=hızlı harekette az gecikme (varsayılan: 0.3)",
    )
    args = parser.parse_args()

    stream = StarLiveStream(
        camera_ids=args.cameras,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        smooth_cutoff=args.smooth_cutoff,
        smooth_beta=args.smooth_beta,
    )
    stream.run(preview=args.preview)


if __name__ == "__main__":
    main()
