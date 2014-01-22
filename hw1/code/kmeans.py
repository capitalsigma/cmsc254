#!/usr/bin/python3

from sys import argv
from random import choice
from PIL import Image
from sys import exit, version_info

if version_info < (3, 0):
    raise Exception("Python >= 3.0 is required.")

def tiff_to_array(image):
    return [[image.getpixel((x, y)) for y in range(0, image.size[0])]
            for x in range(0, image.size[1])]

def overwrite_tiff_with_array(image, points):
    for x, row in enumerate(points):
        for y, point in enumerate(row):
            image.putpixel((x, y), point)
    return image

class ClusterCenter:
    def __init__(self, point):
        self.point = point
        self.count = 0

    def __repr__(self):
        return "point: {}, count: {}".format(self.point, self.count)

    def add(self, point):
        self.point = tuple([sum(x) for x in zip(self.point, point)])
        self.count += 1

        return self

    def avg(self):
        try:
            self.point = tuple([p/self.count for p in self.point])
        except ZeroDivisionError:
            self.point = tuple([float("inf") for p in self.point])
        self.count = 0

        return self

class Kmeans:
    MAX_ITERATIONS = 30
    def __init__(self, points, n_centers):
        self.points = points
        self.n_centers = n_centers
        assert(points)
        assert(n_centers)

    @staticmethod
    def _dist(p1, p2):
        assert(len(p1) == len(p2))
        return sum([(p2[a] - p1[a]) ** 2 for a in range(0, len(p1))])

    @classmethod
    def _nearest(cls, point, cluster_centers):
        assert(point)
        assert(cluster_centers)
        return min(enumerate([cls._dist(point, cc.point)
                              for cc in cluster_centers]),
                   key=lambda x: x[1])[0]

    @classmethod
    def _nearest_cc(cls, point, cluster_centers):
        return cluster_centers[cls._nearest(point, cluster_centers)]

    def _work(self, old_cluster_centers):
        assert(old_cluster_centers)
        new_cluster_centers = [ClusterCenter((0,) * len(self.points[0][0])) for
                               _ in range(0, self.n_centers)]
        for row in self.points:
            for point in row:
                index = self._nearest(point, old_cluster_centers)
                new_cluster_centers[index].add(point)
                # self._nearest_cc(point, old_cluster_centers).add(point)

        return [c.avg() for c in new_cluster_centers]

    def _update_points(self, final_ccs):
        return [[cc.point for cc in 
                 [self._nearest_cc(p, final_ccs) for p in row]]
                for row in self.points]

    @staticmethod
    def _to_ints(points):
        return [[tuple([int(x) for x in p]) for p in row]
                for row in points]

    def execute(self):
        ccs = [ClusterCenter(choice(choice(self.points)))
               for n in range(0, self.n_centers)]
        for _ in range(0, self.MAX_ITERATIONS):
            new_ccs = self._work(ccs)
            if new_ccs == ccs:
                break
            ccs = new_ccs
            
        final_points = self._update_points(ccs)

        return self._to_ints(final_points)

class VectorQuantize:
    def __init__(self, image, n_centers):
        self.old_image = image
        self.new_image = None
        self.n_centers = n_centers

    def execute(self):
        tiff_array = tiff_to_array(self.old_image)
        new_rgbs = Kmeans(tiff_array, self.n_centers).execute()
        self.new_image = overwrite_tiff_with_array(self.old_image.copy(),
                                                   new_rgbs)

        return self.new_image

    def show(self, show_new=True):
        (self.new_image if show_new else self.old_image).show()
    
    def write(self, path, fmt=None):
        self.new_image.save(path, fmt)


def main():
    USAGE = "{} path_in n_centers [path_out [fmt1]]".format(argv[0])
    try:
        path, n_centers = (argv[1], int(argv[2]))
    except (ValueError, IndexError):
        print(USAGE)
        exit(1)

    vc = VectorQuantize(Image.open(path), int(n_centers))
    vc.execute()

    if(len(argv) > 3):
        vc.write(argv[3], argv[4] if len(argv) == 5 else None)
    else:
        vc.show()

if __name__ == '__main__':
    main()
        
