import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL.ImageDraw import ImageDraw
from itertools import chain

sys.path.append("linedraw")
from linedraw.scripts.perlin import *
from linedraw.scripts.filters import *
from linedraw.scripts.strokesort import *
from linedraw.scripts.util import *


# sys.path.append("linedraw")
# from scripts.perlin import *
# from scripts.filters import *
# from scripts.strokesort import *
# from scripts.util import *
# from itertools import chain


def min_max_list_tuples(nums):
    result_max = map(max, zip(*nums))
    result_min = map(min, zip(*nums))
    return list(result_min), list(result_max)


class LineDraw:
    def __init__(self, export_path: Path = Path("../output/out.svg"),
                 draw_contours: bool = True,
                 draw_hatch: bool = True,
                 hatch_size: int = 16,
                 contour_simplify: int = 2,
                 no_polylines: bool = True,
                 draw_border: bool = True,
                 resize: bool = False,
                 fixed_size: bool = False,
                 longest: int = 110,  # mm
                 shortest: int = 80,  # mm,
                 offset: (float, float) = (0.0, 0.0),
                 resolution: int = 1024):
        """
        :param export_path:
        :param draw_contours:
        :param draw_hatch:
        :param hatch_size:
        :param contour_simplify:
        :param no_polylines: if no_polylines is true, convert all polylines to paths
        :param resolution:
        :param shortest: the shortest side of frame or paper to scale towards
        :param longest: the longest side of frame or paper to scale towards
        ;param offset: offset in mm for translating the drawing
        @param export_path:
        @param draw_contours:
        @param draw_hatch:
        @param hatch_size:
        @param contour_simplify:
        @param no_polylines:
        @param draw_border:
        @param resize:
        @param longest:
        @param shortest:
        @param resolution:
        """
        try:
            import numpy as np
            import cv2
            self.no_cv = False
        except ImportError as e:
            print("Cannot import numpy/openCV. Switching to NO_CV mode. error: {}".format(e))
            self.no_cv = True
        self.export_path = export_path
        self.draw_contours = draw_contours
        self.draw_hatch = draw_hatch
        self.show_bitmap = False
        self.resolution = resolution
        self.hatch_size = hatch_size
        self.contour_simplify = contour_simplify
        self.perlin = Perlin()
        self.no_polylines = no_polylines
        self.border = draw_border

        self.resize = resize
        self.orig_width = 0
        self.orig_heigth = 0
        self.center_transformation = (0, 0)
        self.fixed_size = fixed_size
        self.longest = longest
        self.shortest = shortest
        self.x_min = 0
        self.y_min = 0
        self.x_max = self.shortest * 3.7795
        self.y_max = self.longest * 3.7795
        self.scale_factor = 1.0  # scaling factor to be calculated to have the image at max within the borders
        self.scale_margin_factor = 0.9  # factor so a drawing is a little of the edges (visually more pleasing)
        self.offset = offset

    def find_edges(self, IM):
        print("finding edges...")
        if self.no_cv:
            # appmask(IM,[F_Blur])
            appmask(IM, [F_SobelX, F_SobelY])
        else:
            im = np.array(IM)
            im = cv2.GaussianBlur(im, (3, 3), 0)
            im = cv2.Canny(im, 100, 200)
            IM = Image.fromarray(im)
        return IM.point(lambda p: p > 128 and 255)

    @staticmethod
    def getdots(IM):
        print("getting contour points...")
        PX = IM.load()
        dots = []
        w, h = IM.size
        for y in range(h - 1):
            row = []
            for x in range(1, w):
                if PX[x, y] == 255:
                    if len(row) > 0:
                        if x - row[-1][0] == row[-1][-1] + 1:
                            row[-1] = (row[-1][0], row[-1][-1] + 1)
                        else:
                            row.append((x, 0))
                    else:
                        row.append((x, 0))
            dots.append(row)
        return dots

    @staticmethod
    def connectdots(dots):
        print("connecting contour points...")
        contours = []
        for y in range(len(dots)):
            for x, v in dots[y]:
                if v > -1:
                    if y == 0:
                        contours.append([(x, y)])
                    else:
                        closest = -1
                        cdist = 100
                        for x0, v0 in dots[y - 1]:
                            if abs(x0 - x) < cdist:
                                cdist = abs(x0 - x)
                                closest = x0

                        if cdist > 3:
                            contours.append([(x, y)])
                        else:
                            found = 0
                            for i in range(len(contours)):
                                if contours[i][-1] == (closest, y - 1):
                                    contours[i].append((x, y,))
                                    found = 1
                                    break
                            if found == 0:
                                contours.append([(x, y)])
            for c in contours:
                if c[-1][1] < y - 1 and len(c) < 4:
                    contours.remove(c)
        return contours

    def getcontours(self, IM, sc=2):
        print("generating contours...")
        IM = self.find_edges(IM)
        IM1 = IM.copy()
        IM2 = IM.rotate(-90, expand=True).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        dots1 = self.getdots(IM1)
        contours1 = self.connectdots(dots1)
        dots2 = self.getdots(IM2)
        contours2 = self.connectdots(dots2)

        for i in range(len(contours2)):
            contours2[i] = [(c[1], c[0]) for c in contours2[i]]
        contours = contours1 + contours2

        for i in range(len(contours)):
            for j in range(len(contours)):
                if len(contours[i]) > 0 and len(contours[j]) > 0:
                    if distsum(contours[j][0], contours[i][-1]) < 8:
                        contours[i] = contours[i] + contours[j]
                        contours[j] = []

        for i in range(len(contours)):
            contours[i] = [contours[i][j] for j in range(0, len(contours[i]), 8)]

        contours = [c for c in contours if len(c) > 1]

        for i in range(0, len(contours)):
            contours[i] = [(v[0] * sc, v[1] * sc) for v in contours[i]]

        for i in range(0, len(contours)):
            for j in range(0, len(contours[i])):
                contours[i][j] = int(contours[i][j][0] + 10 * self.perlin.noise(i * 0.5, j * 0.1, 1)), int(
                    contours[i][j][1] + 10 * self.perlin.noise(i * 0.5, j * 0.1, 2))

        return contours

    def hatch(self, IM, sc=16):
        print("hatching...")
        PX = IM.load()
        w, h = IM.size
        lg1 = []
        lg2 = []
        for x0 in range(w):
            for y0 in range(h):
                x = x0 * sc
                y = y0 * sc
                if PX[x0, y0] > 144:
                    pass

                elif PX[x0, y0] > 64:
                    lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
                elif PX[x0, y0] > 16:
                    lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
                    lg2.append([(x + sc, y), (x, y + sc)])

                else:
                    lg1.append([(x, y + sc / 4), (x + sc, y + sc / 4)])
                    lg1.append([(x, y + sc / 2 + sc / 4), (x + sc, y + sc / 2 + sc / 4)])
                    lg2.append([(x + sc, y), (x, y + sc)])

        lines = [lg1, lg2]
        for k in range(0, len(lines)):
            for i in range(0, len(lines[k])):
                for j in range(0, len(lines[k])):
                    if lines[k][i] != [] and lines[k][j] != []:
                        if lines[k][i][-1] == lines[k][j][0]:
                            lines[k][i] = lines[k][i] + lines[k][j][1:]
                            lines[k][j] = []
            lines[k] = [l for l in lines[k] if len(l) > 0]
        lines = lines[0] + lines[1]

        for i in range(0, len(lines)):
            for j in range(0, len(lines[i])):
                lines[i][j] = int(lines[i][j][0] + sc * self.perlin.noise(i * 0.5, j * 0.1, 1)), int(
                    lines[i][j][1] + sc * self.perlin.noise(i * 0.5, j * 0.1, 2)) - j
        return lines

    # @staticmethod
    def makesvg(self, lines, no_polyline: bool = False):
        print("generating svg file...")
        if self.fixed_size:
            print("linedrawing has a fixed size")
            out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1" width="{}mm" height="{}mm">'.format(
                self.shortest,
                self.longest)
        else:
            out = '<svg xmlns="http://www.w3.org/2000/svg" version="1.1">'
        if self.border:
            out += '<rect x="0" width="{}mm" height="{}mm" fill="none" stroke="black" stroke-width="1.0"/>'.format(
                self.shortest, self.longest)
        if self.resize:
            print("resizing & recentering")
            self.calc_scale_factor(lines)
            self.recenter()
            out += '<g transform="scale( {} {}) translate( {} {} )">'.format(
                self.scale_factor * self.scale_margin_factor,
                self.scale_factor * self.scale_margin_factor,
                self.center_transformation[0] + self.offset[0] * 3.7795,
                self.center_transformation[1] + self.offset[1] * 3.7795)
        if not no_polyline:
            line_type_prefix = '<polyline points="'
        else:
            print("we're using paths instead of polylines")
            line_type_prefix = '<path d="M'
        for l in lines:
            l = ",".join([str(p[0]) + "," + str(p[1]) for p in l])
            out += line_type_prefix + l + '" stroke="black" stroke-width="0.1" fill="none" />\n'
        if self.resize:
            out += '</g>'
        out += '</svg>'
        return out

    def calc_scale_factor(self, lines):
        # to find the max width and height, we first flatten the 2D array
        flattened_lines = list(chain.from_iterable(lines))
        # then we look for the max x and max y in all tuples
        (self.x_min, self.y_min), (self.x_max, self.y_max) = min_max_list_tuples(flattened_lines)
        self.orig_width = self.x_max - self.x_min
        self.orig_heigth = self.y_max - self.y_min
        # we have to take the conversion mm to px into account: 1mm ≅ 3.7795px or
        # user units [oreilly](https://oreillymedia.github.io/Using_SVG/guide/units.html)
        scale_longest = self.longest * 3.7795 / self.orig_heigth
        scale_shortest = self.shortest * 3.7795 / self.orig_width
        # we take the smallest scale factor to scale both directions
        self.scale_factor = min(scale_longest, scale_shortest)
        print("scale factor = {}".format(self.scale_factor))

    def recenter(self):
        old_center = (round((self.orig_width / 2 + self.x_min)),
                      round((self.orig_heigth / 2 + self.y_min)))
        new_center = (round(self.shortest * 3.7795 / (2 * self.scale_factor * self.scale_margin_factor)),
                      round(self.longest * 3.7795 / (2 * self.scale_factor * self.scale_margin_factor)))
        self.center_transformation = tuple(np.subtract(new_center, old_center))

    def sketch(self, path):
        IM = None
        possible = [path, "images/" + path, "images/" + path + ".jpg", "images/" + path + ".png",
                    "images/" + path + ".tif"]
        for p in possible:
            try:
                IM = Image.open(p)
                break
            except FileNotFoundError:
                print("The Input File wasn't found. Check Path")
                exit(0)
                pass
        w, h = IM.size

        IM = IM.convert("L")
        IM = ImageOps.autocontrast(IM, 10)

        lines = []
        if self.draw_contours:
            lines += self.getcontours(
                IM.resize(
                    (self.resolution // self.contour_simplify, self.resolution // self.contour_simplify * h // w)),
                self.contour_simplify)
        if self.draw_hatch:
            lines += self.hatch(
                IM.resize((self.resolution // self.hatch_size, self.resolution // self.hatch_size * h // w)),
                self.hatch_size)

        lines = sortlines(lines)
        if self.show_bitmap:
            disp = Image.new("RGB", (self.resolution, self.resolution * h // w), (255, 255, 255))
            draw = ImageDraw.Draw(disp)
            for l in lines:
                draw.line(l, (0, 0, 0), 5)
            disp.show()
        # if self.resize:
        #     lines = self.resize_svg_lines(lines)
        svg = self.makesvg(lines, self.no_polylines)
        with open(self.export_path, 'w') as f:
            f.write(svg)
        print(len(lines), "strokes.")
        print("done.")
        return lines


if __name__ == "__main__":
    ld = LineDraw()

    parser = argparse.ArgumentParser(description='Convert image to vectorized line drawing for plotters.')
    parser.add_argument('-i', '--input', dest='input_path',
                        default='lenna', action='store', nargs='?', type=str,
                        help='Input path')

    parser.add_argument('-o', '--output', dest='output_path',
                        default=ld.export_path, action='store', nargs='?', type=str,
                        help='Output path.')

    parser.add_argument('-b', '--show_bitmap', dest='show_bitmap',
                        const=not ld.show_bitmap, default=ld.show_bitmap, action='store_const',
                        help="Display bitmap preview.")

    parser.add_argument('-nc', '--no_contour', dest='no_contour',
                        const=ld.draw_contours, default=not ld.draw_contours, action='store_const',
                        help="Don't draw contours.")

    parser.add_argument('-nh', '--no_hatch', dest='no_hatch',
                        const=ld.draw_hatch, default=not ld.draw_hatch, action='store_const',
                        help='Disable hatching.')

    parser.add_argument('--no_cv', dest='no_cv',
                        const=not ld.no_cv, default=ld.no_cv, action='store_const',
                        help="Don't use openCV.")

    parser.add_argument('--hatch_size', dest='hatch_size',
                        default=ld.hatch_size, action='store', nargs='?', type=int,
                        help='Patch size of hatches. eg. 8, 16, 32')
    parser.add_argument('--contour_simplify', dest='contour_simplify',
                        default=ld.contour_simplify, action='store', nargs='?', type=int,
                        help='Level of contour simplification. eg. 1, 2, 3')
    parser.add_argument('--no_polylines', '-np', dest='no_polylines', const=ld.no_polylines,
                        default=not ld.no_polylines,
                        action='store_const', help="Don't use polylines.")
    parser.add_argument('--draw-border', '-db', dest='draw_border', const=ld.border,
                        default=not ld.border,
                        action='store_const', help="draw border")
    parser.add_argument('--resize', '-rs', dest='resize', const=not ld.resize,
                        default=ld.resize,
                        action='store_const', help="resize")
    parser.add_argument('--fixed_size', '-fs', dest='fixed_size',
                        const=not ld.fixed_size, default=ld.fixed_size, action='store_const',
                        help='fixate size of the paper (mm)')
    parser.add_argument('--longest', '-x', dest='longest',
                        default=110, action='store', nargs='?', type=int,
                        help='longest side of paper (mm)')
    parser.add_argument('--shortest', '-y', dest='shortest',
                        default=80, action='store', nargs='?', type=int,
                        help='shortest side of paper (mm)')
    parser.add_argument('--offset', '-of', dest='offset',
                        default=(0, 0), action='store', nargs=2, type=float,
                        help='offset of the paper (mm)')
    args = parser.parse_args()

    ld.export_path = args.output_path
    ld.draw_hatch = not args.no_hatch
    ld.draw_contours = not args.no_contour
    ld.hatch_size = args.hatch_size
    ld.contour_simplify = args.contour_simplify
    ld.show_bitmap = args.show_bitmap
    ld.no_cv = args.no_cv
    ld.no_polylines = args.no_polylines
    ld.resize = args.resize
    ld.longest = args.longest
    ld.shortest = args.shortest
    ld.fixed_size = args.fixed_size
    ld.border = args.draw_border
    ld.offset = tuple(args.offset)
    ld.sketch(args.input_path)
