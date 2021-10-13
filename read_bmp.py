import struct
import sys
import cv2 as cv

args = sys.argv
print(args)
fn = args[1]

img = cv.imread(fn, -1)
print(img.shape)

bmp = open(fn, 'rb')
print('Type:', bmp.read(2).decode())
print('Size: %s' % struct.unpack('I', bmp.read(4)))
print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
print('Offset: %s' % struct.unpack('I', bmp.read(4)))

print('DIB Header Size: %s' % struct.unpack('I', bmp.read(4)))
print('Width: %s' % struct.unpack('I', bmp.read(4)))
print('Height: %s' % struct.unpack('I', bmp.read(4)))
print('Colour Planes: %s' % struct.unpack('H', bmp.read(2)))
print('Bits<Plug>PeepOpener Pixel: %s' % struct.unpack('H', bmp.read(2)))
print('Compression Method: %s' % struct.unpack('I', bmp.read(4)))
print('Raw Image Size: %s' % struct.unpack('I', bmp.read(4)))
print('Horizontal Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Vertical Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Number of Colours: %s' % struct.unpack('I', bmp.read(4)))
print('Important Colours: %s' % struct.unpack('I', bmp.read(4)))
