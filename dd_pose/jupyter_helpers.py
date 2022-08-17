import PIL.Image
import IPython.display


def showimage(a):
    """Show an image below the current jupyter notebook cell.
    Expects gray or bgr input (opencv2 default)"""
    # bgr -> rgb
    if len(a.shape) > 2 and a.shape[2] == 3:
        a = a[..., ::-1]  # bgr -> rgb
    image = PIL.Image.fromarray(a)
    IPython.display.display(image)  # display in cell output
