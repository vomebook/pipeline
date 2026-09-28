"""Explicit local Poppler acceptance: no model download and no network."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from scripts import pdf_ocr, pdf_ocr_stages as stages


class RealPdfRenderingTests(unittest.TestCase):
    def test_small_scanned_pdf_renders_lossless_ocr_input_before_recognition(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "small.pdf"
            with Image.new("RGB", (300, 400), "white") as image:
                ImageDraw.Draw(image).text((30, 50), "OCR input acceptance", fill="black")
                image.save(source, "PDF", resolution=150)
            self.assertLess(source.stat().st_size, 100 * 1024 ** 2)
            bundle = root / "bundle"
            with patch.object(pdf_ocr, "ocr_page", side_effect=AssertionError("rendering must not OCR")):
                result = stages.render_book({"key": "test\0small.pdf", "source_revision": "test"}, source, bundle)
            manifest = json.loads((bundle / result["render_manifest"]["path"]).read_text())
            stages.validate_render(result, manifest)
            self.assertEqual(len(manifest["pages"]), 1)
            page = manifest["pages"][0]
            self.assertEqual(page["source"], "ocr")
            with Image.open(bundle / page["i"]) as png:
                self.assertEqual(png.format, "PNG")
                self.assertEqual(png.size, (600, 800))
            with Image.open(bundle / page["w"]) as webp:
                self.assertEqual(webp.format, "WEBP")
            self.assertNotIn("o", page)

    def test_low_resolution_full_page_scan_preserves_ocr_pixels_but_not_reader_upscale(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "low-res.pdf"
            with Image.new("RGB", (300, 400), "white") as image:
                ImageDraw.Draw(image).text((30, 50), "Source pixels", fill="black")
                image.save(source, "PDF", resolution=100)
            self.assertEqual(pdf_ocr.scan_reader_images(source, 1, 1), {1: (300, 400)})
            def encode(reader_png, destination):
                with Image.open(reader_png) as image:
                    self.assertLessEqual(image.width, 300)
                    self.assertLessEqual(image.height, 400)
                destination.write_bytes(b"jxl")
            with patch.object(pdf_ocr, "JXL_ENABLED", True), patch.object(pdf_ocr, "encode_jxl", side_effect=encode):
                bundle = root / "bundle"
                result = stages.render_book({"key": "test\0low-res.pdf", "source_revision": "test"}, source, bundle)
            page = json.loads((bundle / result["render_manifest"]["path"]).read_text())["pages"][0]
            with Image.open(bundle / page["i"]) as png, Image.open(bundle / page["w"]) as webp:
                self.assertGreater(png.width, 300)
                self.assertGreater(png.height, 400)
                self.assertLessEqual(webp.width, 300)
                self.assertLessEqual(webp.height, 400)
            self.assertEqual((bundle / page["j"]).read_bytes(), b"jxl")

    def test_small_embedded_image_is_not_treated_as_full_page_scan(self):
        listing = ("page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio\n"
                   "----\n 1 0 image 300 400 rgb 3 8 jpeg no 1 0 100 100 3000B 1%")
        def inspect(command):
            return listing if command[0] == "pdfimages" else "Page 1 size: 612 x 792 pts\n"
        with patch.object(pdf_ocr, "_run", side_effect=inspect):
            self.assertEqual(pdf_ocr.scan_reader_images(Path("sample.pdf"), 1, 1), {})

    def test_reader_jxl_and_webp_share_the_reading_cap_without_changing_ocr_png(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "large-page.pdf"
            with Image.new("RGB", (800, 1100), "white") as image:
                image.save(source, "PDF", resolution=150)
            png, width, height = pdf_ocr.render_page(source, 1, root, reader_jxl=True)
            reader_png = png.with_name(png.stem + "-reader.png")
            with Image.open(png) as full, Image.open(reader_png) as reader, \
                    Image.open(png.with_suffix(".webp")) as webp:
                self.assertEqual(full.size, (width, height))
                self.assertGreater(full.height, 1800)
                self.assertEqual(reader.size, webp.size)
                self.assertEqual(reader.height, 1800)

    def test_reader_quality_uses_clean_scan_only_with_or_without_jxl(self):
        with Image.new("RGB", (128, 128), "white") as clean, \
                Image.new("RGB", (128, 128), (196, 166, 115)) as yellowed, \
                Image.new("RGB", (128, 128), "black") as damaged:
            self.assertEqual(pdf_ocr.reader_webp_quality(clean, True), 80)
            self.assertEqual(pdf_ocr.reader_webp_quality(clean, False), 85)
            self.assertEqual(pdf_ocr.reader_webp_quality(yellowed, True), 85)
            self.assertEqual(pdf_ocr.reader_webp_quality(damaged, True), 85)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source = root / "clean.pdf"
                clean.save(source, "PDF", resolution=100)
                for jxl in (False, True):
                    with self.subTest(jxl=jxl), patch.object(pdf_ocr, "reader_webp_quality",
                                                           wraps=pdf_ocr.reader_webp_quality) as select:
                        pdf_ocr.render_page(source, 1, root, (128, 128), reader_jxl=jxl)
                        self.assertEqual(select.call_args.args[1], True)

    def test_batched_pages_match_individual_png_and_webp_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "pages.pdf"
            with Image.new("RGB", (160, 200), "white") as image:
                image.save(source, "PDF", resolution=150, save_all=True,
                           append_images=[Image.new("RGB", (160, 200), color)
                                          for color in ("red", "green", "blue")])
            single, batch = root / "single", root / "batch"
            single.mkdir()
            batch.mkdir()
            for page in range(1, 5):
                pdf_ocr.render_page(source, page, single)
            with patch.object(pdf_ocr, "_run", wraps=pdf_ocr._run) as run:
                self.assertEqual(pdf_ocr.prerender_pages(source, range(1, 5), batch), set(range(1, 5)))
                for page in range(1, 5):
                    pdf_ocr.render_page(source, page, batch, prepared=True)
            self.assertEqual(sum(call.args[0][0] == "pdftocairo" for call in run.call_args_list), 1)
            for page in range(1, 5):
                name = f"page-{page:06d}"
                for suffix in (".png", ".webp"):
                    self.assertEqual((single / (name + suffix)).read_bytes(),
                                     (batch / (name + suffix)).read_bytes())

    def test_incomplete_batch_falls_back_without_stale_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def fail(command, **_kwargs):
                (root / "render-batch-1.png").write_bytes(b"partial")
                raise RuntimeError("batch failed")
            with patch.object(pdf_ocr, "_page_render_dpi", return_value=150), \
                    patch.object(pdf_ocr, "_run", side_effect=fail):
                self.assertEqual(pdf_ocr.prerender_pages(Path("book.pdf"), range(1, 3), root), set())
            self.assertEqual(list(root.iterdir()), [])

    def test_render_book_uses_batch_and_keeps_every_page(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "four.pdf"
            with Image.new("RGB", (150, 200), "white") as image:
                image.save(source, "PDF", resolution=150, save_all=True,
                           append_images=[image, image, image])
            with patch.object(pdf_ocr, "_run", wraps=pdf_ocr._run) as run:
                result = stages.render_book({"key": "test\0four.pdf", "source_revision": "test"},
                                            source, root / "bundle")
            self.assertEqual(sum(call.args[0][0] == "pdftocairo" for call in run.call_args_list), 1)
            manifest = json.loads((root / "bundle" / result["render_manifest"]["path"]).read_text())
            self.assertEqual([page["p"] for page in manifest["pages"]], [1, 2, 3, 4])
            stages.validate_render(result, manifest)

    def test_mixed_dpi_batch_uses_individual_fallback(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(pdf_ocr, "_page_render_dpi", side_effect=(150, 200)), \
                patch.object(pdf_ocr, "_run") as run:
            self.assertEqual(pdf_ocr.prerender_pages(Path("book.pdf"), range(1, 3), Path(temp)), set())
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
