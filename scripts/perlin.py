# Perlin Noise
# Based on Javascript from p5.js (https://github.com/processing/p5.js/blob/master/src/math/noise.js)

import math
import random


class Perlin:
    def __init__(self):
        self.PERLIN_YWRAPB = 4
        self.PERLIN_YWRAP = 1 << self.PERLIN_YWRAPB
        self.PERLIN_ZWRAPB = 8
        self.PERLIN_ZWRAP = 1 << self.PERLIN_ZWRAPB
        self.PERLIN_SIZE = 4095

        self.perlin_octaves = 4
        self.perlin_amp_falloff = 0.5
        self.perlin = None

    def noise(self, x, y=0, z=0):
        # global perlin
        if self.perlin is None:
            self.perlin = []
            for i in range(0, self.PERLIN_SIZE + 1):
                self.perlin.append(random.random())
        if x < 0: x = -x
        if y < 0: y = -y
        if z < 0: z = -z

        xi, yi, zi = int(x), int(y), int(z)
        xf = x - xi
        yf = y - yi
        zf = z - zi
        rxf = ryf = None

        r = 0
        ampl = 0.5

        n1 = n2 = n3 = None
        for o in range(0, self.perlin_octaves):
            of = xi + (yi << self.PERLIN_YWRAPB) + (zi << self.PERLIN_ZWRAPB)

            rxf = scaled_cosine(xf)
            ryf = scaled_cosine(yf)

            n1 = self.perlin[of & self.PERLIN_SIZE]
            n1 += rxf * (self.perlin[(of + 1) & self.PERLIN_SIZE] - n1)
            n2 = self.perlin[(of + self.PERLIN_YWRAP) & self.PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + self.PERLIN_YWRAP + 1) & self.PERLIN_SIZE] - n2)
            n1 += ryf * (n2 - n1)

            of += self.PERLIN_ZWRAP
            n2 = self.perlin[of & self.PERLIN_SIZE]
            n2 += rxf * (self.perlin[(of + 1) & self.PERLIN_SIZE] - n2)
            n3 = self.perlin[(of + self.PERLIN_YWRAP) & self.PERLIN_SIZE]
            n3 += rxf * (self.perlin[(of + self.PERLIN_YWRAP + 1) & self.PERLIN_SIZE] - n3)
            n2 += ryf * (n3 - n2)

            n1 += scaled_cosine(zf) * (n2 - n1)

            r += n1 * ampl
            ampl *= self.perlin_amp_falloff
            xi <<= 1
            xf *= 2
            yi <<= 1
            yf *= 2
            zi <<= 1
            zf *= 2

            if xf >= 1.0: xi += 1; xf -= 1
            if yf >= 1.0: yi += 1; yf -= 1
            if zf >= 1.0: zi += 1; zf -= 1
        return r

    def noiseDetail(self, lod, falloff):
        if lod > 0: self.perlin_octaves = lod
        if falloff > 0: self.perlin_amp_falloff = falloff

    def noiseSeed(self, seed):
        lcg = LCG()
        lcg.setSeed(seed)
        self.perlin = []
        for i in range(0, self.PERLIN_SIZE + 1):
            self.perlin.append(lcg.rand())


class LCG:
    def __init__(self):
        self.m = 4294967296.0
        self.a = 1664525.0
        self.c = 1013904223.0
        self.seed = self.z = None

    def setSeed(self, val=None):
        self.z = self.seed = (random.random() * self.m if val == None else val) >> 0

    def getSeed(self):
        return self.seed

    def rand(self):
        self.z = (self.a * self.z + self.c) % self.m
        return self.z / self.m


def scaled_cosine(i):
    return 0.5 * (1.0 - math.cos(i * math.pi))
