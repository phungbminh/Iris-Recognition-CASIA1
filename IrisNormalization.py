import numpy as np
import warnings
warnings.filterwarnings("ignore")



class IrisNormalization:
    def normalize(self, image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil,
                  radpixels, angulardiv):
        """
        Turn a circular region into a rectangular block of NxN dimensions
        """
        radiuspixels = radpixels + 2
        angledivisions = angulardiv - 1

        r = np.arange(radiuspixels)
        theta = np.linspace(0, 2 * np.pi, angledivisions + 1)

        # displacement of pupil and iris center
        ox = x_pupil - x_iris
        oy = y_pupil - y_iris

        if ox <= 0:
            sgn = -1
        elif ox > 0:
            sgn = 1

        if ox == 0 and oy > 0:
            sgn = 1

        a = np.ones(angledivisions + 1) * (ox ** 2 + oy ** 2)

        if ox == 0:
            phi = np.pi / 2
        else:
            phi = np.arctan(oy / ox)

        b = sgn * np.cos(np.pi - phi - theta)

        r = np.sqrt(a) * b + np.sqrt(a * b ** 2 - (a - r_iris ** 2))
        r = np.array([r - r_pupil])

        rmat = np.dot(np.ones([radiuspixels, 1]), r)

        rmat = rmat * np.dot(np.ones([angledivisions + 1, 1]), np.array([np.linspace(0, 1, radiuspixels)])).transpose()
        rmat = rmat + r_pupil

        # exclude the boundaries of the pupil iris border
        rmat = rmat[1:radiuspixels - 1, :]

        # cartesian location of each point around the iris
        xcosmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.cos(theta)]))
        xsinmat = np.dot(np.ones([radiuspixels - 2, 1]), np.array([np.sin(theta)]))
        xo = rmat * xcosmat
        yo = rmat * xsinmat
        xo = x_pupil + xo
        xo = np.round(xo).astype(int)
        coords = np.where(xo >= image.shape[1])
        xo[coords] = image.shape[1] - 1
        coords = np.where(xo < 0)
        xo[coords] = 0

        yo = y_pupil - yo
        yo = np.round(yo).astype(int)
        coords = np.where(yo >= image.shape[0])
        yo[coords] = image.shape[0] - 1
        coords = np.where(yo < 0)
        yo[coords] = 0

        polar_array = image[yo, xo]
        polar_array = polar_array / 255

        # noise array with location of NaNs in polar_array
        polar_noise = np.zeros(polar_array.shape)
        coords = np.where(np.isnan(polar_array))
        polar_noise[coords] = 1

        # Get rid of outlying points
        image[yo, xo] = 255

        # Get pixel coords for iris
        x, y = self.circlecoords([x_iris, y_iris], r_iris, image.shape)
        image[y, x] = 255

        xp, yp = self.circlecoords([x_pupil, y_pupil], r_pupil, image.shape)
        image[yp, xp] = 255

        # Replace NaNs before performing feature encoding
        coords = np.where((np.isnan(polar_array)))
        polar_array2 = polar_array
        polar_array2[coords] = 0.5
        avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
        polar_array[coords] = avg

        return polar_array, polar_noise.astype(bool)

    def circlecoords(self, c, r, imgsize, nsides=600):
        """
        Find the coordinates of a circle based on its centre and radius.
        """
        a = np.linspace(0, 2 * np.pi, 2 * nsides + 1)
        xd = np.round(r * np.cos(a) + c[0])
        yd = np.round(r * np.sin(a) + c[1])

        # remove value bigger than the image
        xd2 = xd
        coords = np.where(xd >= imgsize[1])
        xd2[coords[0]] = imgsize[1] - 1
        coords = np.where(xd < 0)
        xd2[coords[0]] = 0

        yd2 = yd
        coords = np.where(yd >= imgsize[0])
        yd2[coords[0]] = imgsize[0] - 1
        coords = np.where(yd < 0)
        yd2[coords[0]] = 0

        x = np.round(xd2).astype(int)
        y = np.round(yd2).astype(int)
        return x, y