from sys import argv
from random import choice
from PIL import Image

def tiff_to_array(image):
    return [[image.getpixel((x, y)) for x in range(0, image.size[0])]
            for y in range(0, image.size[1])]

def overwrite_tiff_with_array(image, points):
    for row in points:
        for point in row:
            image.setpixel(point)
    return image

class ClusterCenter:
    def __init__(self, point):
        self.point = point
        self.count = 0

    def add(self, point):
        self.point = tuple([sum(x) for x in zip(self.point, point)])
        self.count += 1

    def avg(self):
        try:
            self.point = tuple([p/self.count for p in self.point])
        except ZeroDivisionError:
            self.point = tuple([float("inf") for p in self.point])

class Kmeans:
    MAX_ITERATIONS = 30
    def __init__(self, points, n_centers):
        self.points = points
        self.n_centers = n_centers

    @staticmethod
    def _dist(p1, p2):
        assert(len(p1) == len(p2))
        return sum([(p2[a] - p1[a]) ** 2 for a in range(0, len(p1))])

    def _nearest(self, point, cluster_centers):
        return min(enumerate([self._dist(point, cc.point)
                              for cc in cluster_centers]),
                   key=lambda x: x[1])

    def _work(self, old_cluster_centers):
        new_cluster_centers = [ClusterCenter((0,) * len(self.points[0][0])) for
                               _ in range(0, self.n_centers)]
        for row in self.points:
            for point in row:
                index = self._nearest(point, old_cluster_centers)
                new_cluster_centers[index].add(point)

        return [c.avg() for c in new_cluster_centers]

    def execute(self):
        ccs = [ClusterCenter(choice(choice(self.points)))
               for n in range(0, self.n_centers)]
        for _ in range(0, self.MAX_ITERATIONS):
            new_ccs = self._work(ccs)
            if new_ccs == ccs:
                break
            ccs = new_ccs

        return ccs

class VectorQuantize:
    def __init__(self, image):
        self.old_image = image
        self.new_image = None

    def execute(self):
        tiff_array = tiff_to_array(self.old_image)
        new_rgbs = Kmeans(tiff_array).execute()
        self.new_image = overwrite_tiff_with_array(self.old_image.copy(),
                                                   new_rgbs)

        return self.new_image

    def show(self, show_new=True):
        (self.new_image if show_new else self.old_image).show()

def main(path):
    vc = VectorQuantize(Image.open(path))
    vc.execute()
    vc.show()

if __name__ == '__main__':
    main(argv[1])
