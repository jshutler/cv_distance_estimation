from math import pi

import decimal

def float_range(start, stop, step):
	while start < stop:
		yield float(start)
		start += step


step = pi/8




bins = [(angle % (2*pi), (angle + pi/4) %(pi*2)) for angle in float_range(pi/8, (15*pi / 8) + pi/8, pi/4)]

bins2 = [(angle, (angle + pi/4)) for angle in float_range(pi/8, (15*pi / 8) + pi/8, pi/4)]


# bins = [((step*i) % (2*pi), (step*(i+1)) % (2*pi)) for i in range(1, 9)]

print(bins)


for i in range(8):
	print(i, bins[i], bins2[i])

