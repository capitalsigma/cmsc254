import unittest 
from kmeans import *

class TestGlobalFuncs(unittest.TestCase):
    def test_to_and_from(self):
        old = Image.open('../input/mandrill-small.tiff')
        array = tiff_to_array(old)
        new = overwrite_tiff_with_array(old.copy(), array)

        self.assertEqual(new.size, old.size)
        for pair in [(x, y) for x in range(0, old.size[0])
                     for y in range(0, old.size[1])]:
                self.assertEqual(old.getpixel(pair), new.getpixel(pair))


class TestClusterCenter(unittest.TestCase):
    def setUp(self):
        # self.point = [[(1, 2), (3, 4)], [(5, 6), (7, 8)]]
        self.blank = (0, 0)
        self.ones = (1, 1)
        self.cc_blank = ClusterCenter(self.blank)
        self.cc_ones = ClusterCenter(self.ones)

    def test_add(self):
        self.cc_blank.add(self.blank)
        self.assertEqual(self.cc_blank.point, self.blank)

        self.cc_blank.add(self.ones)
        self.assertEqual(self.cc_blank.point, self.ones)
        self.assertEqual(self.cc_blank.count, 2)

    def test_avg(self):
        for _ in range(0, 11):
            self.cc_blank.add(self.ones)

        self.cc_blank.avg()
        self.assertEqual(self.cc_blank.point, self.ones)
        self.assertEqual(self.cc_blank.count, 0)


class TestKmeans(unittest.TestCase):
    def setUp(self):
        ps = lambda x: [[(x, x) * 10] * 10]
        self.zero = (0, 0)
        self.one = (1, 1)
        self.zeros = ps(0)
        self.ones = ps(1)
        self.km_zeros = Kmeans(self.zeros, 1)
        self.km_ones = Kmeans(self.ones, 5)
        self.cc_blanks = [ClusterCenter(self.zero)] * 3
        self.cc_mixed = [ClusterCenter(self.zero), ClusterCenter(self.one)]
        
    def test_init(self):
        self.assertEqual(self.km_zeros.points, self.zeros)
        self.assertEqual(self.km_ones.points, self.ones)
        self.assertEqual(self.km_zeros.n_centers, 1)
        self.assertEqual(self.km_ones.n_centers, 5)        
        

    def test_dist(self):
        unit_dist = Kmeans._dist(self.one, self.zero)
        zero_dist = Kmeans._dist(self.zero, self.zero)
        zero2 = Kmeans._dist(self.one, self.one)
        
        
        self.assertEqual(unit_dist, 2)
        self.assertEqual(zero_dist, 0)
        self.assertEqual(zero2, 0)

    def test_nearest(self):
        nearest_0 = Kmeans._nearest(self.zero, self.cc_mixed)
        nearest_1 = Kmeans._nearest(self.one, self.cc_mixed)
        
        self.assertEqual(nearest_0, 0)
        self.assertEqual(nearest_1, 1)

    def test_work(self):
        pass
        


if __name__ == '__main__':
    unittest.main()
        
        
