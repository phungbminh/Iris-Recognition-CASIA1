import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import multiprocessing as mp
from skimage.transform import radon
from scipy.ndimage import convolve

class SegmentIris:
    def searchInnerBound(self, img):
        """
        Searching of the boundary (inner) of the iris
        """
        # integro-differential
        Y = img.shape[0]
        X = img.shape[1]
        sect = X / 4
        minrad = 10
        maxrad = sect * 0.8
        jump = 4  # Precision of the search

        # Hough Space
        sz = np.array([np.floor((Y - 2 * sect) / jump),
                       np.floor((X - 2 * sect) / jump),
                       np.floor((maxrad - minrad) / jump)]).astype(int)

        # circular integration
        integrationprecision = 1
        angs = np.arange(0, 2 * np.pi, integrationprecision)
        x, y, r = np.meshgrid(np.arange(sz[1]),
                              np.arange(sz[0]),
                              np.arange(sz[2]))
        y = sect + y * jump
        x = sect + x * jump
        r = minrad + r * jump
        hs = self.ContourIntegralCircular(img, y, x, r, angs)

        # Hough Space Partial Derivative
        hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

        # blurring the image
        sm = 3
        hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

        indmax = np.argmax(hspdrs.ravel())
        y, x, r = np.unravel_index(indmax, hspdrs.shape)

        inner_y = sect + y * jump
        inner_x = sect + x * jump
        inner_r = minrad + (r - 1) * jump

        # Integro-Differential
        integrationprecision = 0.1
        angs = np.arange(0, 2 * np.pi, integrationprecision)
        x, y, r = np.meshgrid(np.arange(jump * 2),
                              np.arange(jump * 2),
                              np.arange(jump * 2))
        y = inner_y - jump + y
        x = inner_x - jump + x
        r = inner_r - jump + r
        hs = self.ContourIntegralCircular(img, y, x, r, angs)

        # Hough Space Partial Derivative
        hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

        # blurring the image
        sm = 3
        hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")
        indmax = np.argmax(hspdrs.ravel())
        y, x, r = np.unravel_index(indmax, hspdrs.shape)

        inner_y = inner_y - jump + y
        inner_x = inner_x - jump + x
        inner_r = inner_r - jump + r - 1

        return inner_y, inner_x, inner_r

    def searchOuterBound(self, img, inner_y, inner_x, inner_r):
        """
        Searching of the boundary (outer) of the iris
        """
        maxdispl = np.round(inner_r * 0.15).astype(int)
        minrad = np.round(inner_r / 0.8).astype(int)
        maxrad = np.round(inner_r / 0.3).astype(int)

        # Integration region and avoiding eyelids
        intreg = np.array([[2 / 6, 4 / 6], [8 / 6, 10 / 6]]) * np.pi

        # circular integration
        integrationprecision = 0.05
        angs = np.concatenate([np.arange(intreg[0, 0], intreg[0, 1], integrationprecision),
                               np.arange(intreg[1, 0], intreg[1, 1], integrationprecision)],
                              axis=0)
        x, y, r = np.meshgrid(np.arange(2 * maxdispl),
                              np.arange(2 * maxdispl),
                              np.arange(maxrad - minrad))
        y = inner_y - maxdispl + y
        x = inner_x - maxdispl + x
        r = minrad + r
        hs = self.ContourIntegralCircular(img, y, x, r, angs)

        # Hough Space Partial Derivative
        hspdr = hs - hs[:, :, np.insert(np.arange(hs.shape[2] - 1), 0, 0)]

        # blurring
        sm = 7  # Size of the blurring mask
        hspdrs = signal.fftconvolve(hspdr, np.ones([sm, sm, sm]), mode="same")

        indmax = np.argmax(hspdrs.ravel())
        y, x, r = np.unravel_index(indmax, hspdrs.shape)

        outer_y = inner_y - maxdispl + y + 1
        outer_x = inner_x - maxdispl + x + 1
        outer_r = minrad + r - 1

        return outer_y, outer_x, outer_r

    def ContourIntegralCircular(self, imagen, y_0, x_0, r, angs):
        """
           Contour/circular integral using discrete Riemann
        """
        y = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
        x = np.zeros([len(angs), r.shape[0], r.shape[1], r.shape[2]], dtype=int)
        for i in range(len(angs)):
            ang = angs[i]
            y[i, :, :, :] = np.round(y_0 - np.cos(ang) * r).astype(int)
            x[i, :, :, :] = np.round(x_0 + np.sin(ang) * r).astype(int)

            # adapt x and y
        ind = np.where(y < 0)
        y[ind] = 0
        ind = np.where(y >= imagen.shape[0])
        y[ind] = imagen.shape[0] - 1
        ind = np.where(x < 0)
        x[ind] = 0
        ind = np.where(x >= imagen.shape[1])
        x[ind] = imagen.shape[1] - 1

        hs = imagen[y, x]
        hs = np.sum(hs, axis=0)
        return hs.astype(float)

    def plot_iris_boundaries(self, eye_image, inner_y, inner_x, inner_r, outer_y, outer_x, outer_r, mask_top,
                             mask_bottom):
        # Create a copy of the image to draw on
        image_with_boundaries = np.copy(eye_image)

        # Draw inner boundary
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(inner_x + inner_r * np.cos(angle))
            y = int(inner_y + inner_r * np.sin(angle))
            if 0 <= x < image_with_boundaries.shape[1] and 0 <= y < image_with_boundaries.shape[0]:
                image_with_boundaries[y, x] = 255  # Set color white for inner boundary

        # Draw outer boundary
        for angle in np.linspace(0, 2 * np.pi, 100):
            x = int(outer_x + outer_r * np.cos(angle))
            y = int(outer_y + outer_r * np.sin(angle))
            if 0 <= x < image_with_boundaries.shape[1] and 0 <= y < image_with_boundaries.shape[0]:
                image_with_boundaries[y, x] = 255  # Set color white for outer boundary
        # Apply eyelid masks to remove eyelid regions
        image_with_boundaries[mask_top > 0] = 0  # Set color black for upper eyelid region
        image_with_boundaries[mask_bottom > 0] = 0  # Set color black for lower eyelid region

        # Display the image
        plt.figure(figsize=(8, 8))
        plt.imshow(image_with_boundaries, cmap='gray')
        plt.title('Iris Boundaries without Eyelids')
        plt.axis('off')
        plt.show()

    def findTopEyelid(self, imsz, imageiris, irl, icl, rowp, rp, ret_top=None):
        """
        Find and mask for the top eyelid region.
        """
        topeyelid = imageiris[0: rowp - irl - rp, :]
        lines = self.findline(topeyelid)
        mask = np.zeros(imsz, dtype=float)

        if lines.size > 0:
            xl, yl = self.linecoords(lines, topeyelid.shape)
            yl = np.round(yl + irl - 1).astype(int)
            xl = np.round(xl + icl - 1).astype(int)

            yla = np.max(yl)
            y2 = np.arange(yla)

            mask[yl, xl] = np.nan
            grid = np.meshgrid(y2, xl)
            mask[grid] = np.nan

        if ret_top is not None:
            ret_top[0] = mask
        return mask

    def findBottomEyelid(self, imsz, imageiris, rowp, rp, irl, icl, ret_bot=None):
        """
        Find and mask for the bottom eyelid region.
        """
        bottomeyelid = imageiris[rowp - irl + rp - 1: imageiris.shape[0], :]
        lines = self.findline(bottomeyelid)
        mask = np.zeros(imsz, dtype=float)

        if lines.size > 0:
            xl, yl = self.linecoords(lines, bottomeyelid.shape)
            yl = np.round(yl + rowp + rp - 3).astype(int)
            xl = np.round(xl + icl - 2).astype(int)
            yla = np.min(yl)
            y2 = np.arange(yla - 1, imsz[0])

            mask[yl, xl] = np.nan
            grid = np.meshgrid(y2, xl)
            mask[grid] = np.nan

        if ret_bot is not None:
            ret_bot[0] = mask
        return mask

    def findline(self, img):
        """
        Find lines in the image using linear Hough transformation and
        Canny detection
        """
        I2, orient = self.canny(img, 2, 0, 1)
        I3 = self.adjgamma(I2, 1.9)
        I4 = self.nonmaxsup(I3, orient, 1.5)
        edgeimage = self.hysthresh(I4, 0.2, 0.15)

        # Radon transformation
        theta = np.arange(180)
        R = radon(edgeimage, theta, circle=False)
        sz = R.shape[0] // 2
        xp = np.arange(-sz, sz + 1, 1)

        maxv = np.max(R)
        if maxv > 25:
            i = np.where(R.ravel() == maxv)
            i = i[0]
        else:
            return np.array([])

        R_vect = R.ravel()
        ind = np.argsort(-R_vect[i])
        u = i.shape[0]
        k = i[ind[0: u]]
        y, x = np.unravel_index(k, R.shape)
        t = -theta[x] * np.pi / 180
        r = xp[y]

        lines = np.vstack([np.cos(t), np.sin(t), -r]).transpose()
        cx = img.shape[1] / 2 - 1
        cy = img.shape[0] / 2 - 1
        lines[:, 2] = lines[:, 2] - lines[:, 0] * cx - lines[:, 1] * cy
        return lines

    def linecoords(self, lines, imsize):
        """
        Find x-, y- coordinates of positions along in a line.
        """
        xd = np.arange(imsize[1])
        yd = (-lines[0, 2] - lines[0, 0] * xd) / lines[0, 1]

        coords = np.where(yd >= imsize[0])
        coords = coords[0]
        yd[coords] = imsize[0] - 1
        coords = np.where(yd < 0)
        coords = coords[0]
        yd[coords] = 0

        x = xd
        y = yd
        return x, y

    def canny(self, im, sigma, vert, horz):
        """
        Canny edge detection.
        """

        def fspecial_gaussian(shape=(3, 3), sig=1):
            m, n = [(ss - 1) / 2 for ss in shape]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            f = np.exp(-(x * x + y * y) / (2 * sig * sig))
            f[f < np.finfo(f.dtype).eps * f.max()] = 0
            sum_f = f.sum()
            if sum_f != 0:
                f /= sum_f
            return f

        hsize = [6 * sigma + 1, 6 * sigma + 1]
        gaussian = fspecial_gaussian(hsize, sigma)
        im = convolve(im, gaussian, mode='constant')
        rows, cols = im.shape

        h = np.concatenate([im[:, 1:cols], np.zeros([rows, 1])], axis=1) - \
            np.concatenate([np.zeros([rows, 1]), im[:, 0: cols - 1]], axis=1)

        v = np.concatenate([im[1: rows, :], np.zeros([1, cols])], axis=0) - \
            np.concatenate([np.zeros([1, cols]), im[0: rows - 1, :]], axis=0)

        d11 = np.concatenate([im[1:rows, 1:cols], np.zeros([rows - 1, 1])], axis=1)
        d11 = np.concatenate([d11, np.zeros([1, cols])], axis=0)
        d12 = np.concatenate([np.zeros([rows - 1, 1]), im[0:rows - 1, 0:cols - 1]], axis=1)
        d12 = np.concatenate([np.zeros([1, cols]), d12], axis=0)
        d1 = d11 - d12

        d21 = np.concatenate([im[0:rows - 1, 1:cols], np.zeros([rows - 1, 1])], axis=1)
        d21 = np.concatenate([np.zeros([1, cols]), d21], axis=0)
        d22 = np.concatenate([np.zeros([rows - 1, 1]), im[1:rows, 0:cols - 1]], axis=1)
        d22 = np.concatenate([d22, np.zeros([1, cols])], axis=0)
        d2 = d21 - d22

        X = (h + (d1 + d2) / 2) * vert
        Y = (v + (d1 - d2) / 2) * horz

        gradient = np.sqrt(X * X + Y * Y)

        orient = np.arctan2(-Y, X)
        neg = orient < 0
        orient = orient * ~neg + (orient + np.pi) * neg
        orient = orient * 180 / np.pi

        return gradient, orient

    def adjgamma(self, im, g):
        """
        Adjust image gamma.
        """
        newim = im
        newim = newim - np.min(newim)
        newim = newim / np.max(newim)
        newim = newim ** (1 / g)
        return newim

    def nonmaxsup(self, in_img, orient, radius):
        """
        Perform non-maxima suppression on an image using an orientation image
        """
        rows, cols = in_img.shape
        im_out = np.zeros([rows, cols])
        iradius = np.ceil(radius).astype(int)

        # precalculatihg x and y offsets to relatives to the center piuxel
        angle = np.arange(181) * np.pi / 180
        xoff = radius * np.cos(angle)
        yoff = radius * np.sin(angle)
        hfrac = xoff - np.floor(xoff)
        vfrac = yoff - np.floor(yoff)
        orient = np.fix(orient)

        # interpolating grey values of the center pixel for the nom maximal suppression
        col, row = np.meshgrid(np.arange(iradius, cols - iradius),
                               np.arange(iradius, rows - iradius))

        ori = orient[row, col].astype(int)
        x = col + xoff[ori]
        y = row - yoff[ori]
        # pixel locations that surround location x,y
        fx = np.floor(x).astype(int)
        cx = np.ceil(x).astype(int)
        fy = np.floor(y).astype(int)
        cy = np.ceil(y).astype(int)
        # integer pixel locations
        bl = in_img[cy, fx]  # bottom left
        br = in_img[cy, cx]  # bottom right
        tl = in_img[fy, fx]  # top left
        tr = in_img[fy, cx]  # top right
        # Bi-linear interpolation for x,y values
        upperavg = tl + hfrac[ori] * (tr - tl)
        loweravg = bl + hfrac[ori] * (br - bl)
        v1 = upperavg + vfrac[ori] * (loweravg - upperavg)

        # same thing but for the other side
        map_candidate_region = in_img[row, col] > v1
        x = col - xoff[ori]
        y = row + yoff[ori]
        fx = np.floor(x).astype(int)
        cx = np.ceil(x).astype(int)
        fy = np.floor(y).astype(int)
        cy = np.ceil(y).astype(int)
        tl = in_img[fy, fx]
        tr = in_img[fy, cx]
        bl = in_img[cy, fx]
        br = in_img[cy, cx]
        upperavg = tl + hfrac[ori] * (tr - tl)
        loweravg = bl + hfrac[ori] * (br - bl)
        v2 = upperavg + vfrac[ori] * (loweravg - upperavg)

        # max local
        map_active = in_img[row, col] > v2
        map_active = map_active * map_candidate_region
        im_out[row, col] = in_img[row, col] * map_active

        return im_out

    def hysthresh(self, im, T1, T2):
        """
        Hysteresis thresholding.
        """
        rows, cols = im.shape
        rc = rows * cols
        rcmr = rc - rows
        rp1 = rows + 1

        bw = im.ravel()  # column vector
        pix = np.where(bw > T1)  # pixels with value > T1
        pix = pix[0]
        npix = pix.size  # pixels with value > T1

        # stack array
        stack = np.zeros(rows * cols)
        stack[0:npix] = pix  # add edge points on the stack
        stp = npix
        for k in range(npix):
            bw[pix[k]] = -1

        O = np.array([-1, 1, -rows - 1, -rows, -rows + 1, rows - 1, rows, rows + 1])

        while stp != 0:  # While the stack is != empty
            v = int(stack[stp - 1])
            stp -= 1

            if rp1 < v < rcmr:  # prevent illegal indices
                index = O + v  # indices of points around this pixel.
                for l in range(8):
                    ind = index[l]
                    if bw[ind] > T2:  # value > T2,
                        stp += 1  # add index onto the stack.
                        stack[stp - 1] = ind
                        bw[ind] = -1

        bw = (bw == -1)  # zero out that was not an edge
        bw = np.reshape(bw, [rows, cols])  # Reshaping the image

        return bw

    def segment(self, eyeim, eyelashes_thres=80, use_multiprocess=True):
        """
        Segment the iris from the image
        """
        # Using Daugman's integro-differential to find the iris boundaries
        rowp, colp, rp = self.searchInnerBound(eyeim)
        row, col, r = self.searchOuterBound(eyeim, rowp, colp, rp)

        # pupil and iris boundaries
        rowp = np.round(rowp).astype(int)
        colp = np.round(colp).astype(int)
        rp = np.round(rp).astype(int)
        row = np.round(row).astype(int)
        col = np.round(col).astype(int)
        r = np.round(r).astype(int)
        cirpupil = [rowp, colp, rp]
        ciriris = [row, col, r]

        # top and bottom eyelid
        imsz = eyeim.shape
        irl = np.round(row - r).astype(int)
        iru = np.round(row + r).astype(int)
        icl = np.round(col - r).astype(int)
        icu = np.round(col + r).astype(int)
        if irl < 0:
            irl = 0
        if icl < 0:
            icl = 0
        if iru >= imsz[0]:
            iru = imsz[0] - 1
        if icu >= imsz[1]:
            icu = imsz[1] - 1
        imageiris = eyeim[irl: iru + 1, icl: icu + 1]

        # using multiprocessing
        if use_multiprocess:
            ret_top = mp.Manager().dict()
            ret_bot = mp.Manager().dict()
            p_top = mp.Process(
                target=self.findTopEyelid,
                args=(imsz, imageiris, irl, icl, rowp, rp, ret_top),
            )
            p_bot = mp.Process(target=self.findBottomEyelid,
                               args=(imsz, imageiris, rowp, rp, irl, icl, ret_bot),
                               )
            p_top.start()
            p_bot.start()
            p_top.join()
            p_bot.join()
            mask_top = ret_top[0]
            mask_bot = ret_bot[0]
        else:
            mask_top = self.findTopEyelid(imsz, imageiris, irl, icl, rowp, rp)
            mask_bot = self.findBottomEyelid(imsz, imageiris, rowp, rp, irl, icl)

            # noise region we mark by NaN value
        imwithnoise = eyeim.astype(float)
        imwithnoise = imwithnoise + mask_top + mask_bot

        # For CASIA dataset, we need to eliminate eyelashes by threshold
        ref = eyeim < eyelashes_thres
        coords = np.where(ref == 1)
        imwithnoise[coords] = np.nan

        return ciriris, cirpupil, imwithnoise