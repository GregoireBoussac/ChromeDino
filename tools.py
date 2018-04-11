import mss
import mss.tools

def screen_capture(top, left, width, height):
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {'top': top, 'left': left, 'width': width, 'height': height}
        output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)

        # Grab the data
        sct_img = sct.grab(monitor)
    return np.array(sct_img)