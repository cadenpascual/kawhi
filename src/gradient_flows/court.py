from matplotlib.patches import Circle, Rectangle, Arc, ConnectionPatch
import numpy as np

# ==== Drawing Functions ===== # 
def draw_court_matplotlib(ax, color='#777777', lw=1.5, half_court=None):
    """Matplotlib version of the high-quality Plotly court with half-court support."""
    
    # 1. Court Perimeter (Fixed doubled line by matching line weight and setting zorder)
    if half_court == 'left':
        ax.add_patch(Rectangle((0, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw))
    elif half_court == 'right':
        ax.add_patch(Rectangle((47, 0), 47, 50, color=color, zorder=0, fill=False, lw=lw))
    else:
        ax.add_patch(Rectangle((0, 0), 94, 50, color=color, zorder=0, fill=False, lw=lw))
    
    # 2. Midcourt line and Circle
    ax.plot([47, 47], [0, 50], color=color, lw=lw, zorder=0)
    if half_court == 'left':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=90, theta2=270, color=color, lw=lw, zorder=0))
    elif half_court == 'right':
        ax.add_patch(Arc((47, 25), 12, 12, theta1=-90, theta2=90, color=color, lw=lw, zorder=0))
    else:
        ax.add_patch(Circle((47, 25), 6, color=color, fill=False, lw=lw, zorder=0))

    # 3. Left Side Features (Fixed 68.3 degree 3pt connection)
    if half_court in [None, 'left']:
        ax.add_patch(Rectangle((0, 17), 19, 16, color=color, fill=False, lw=lw, zorder=0))
        # 3PT Arc mathematically connected to the 14ft corner lines
        ax.add_patch(Arc((5.25, 25), 47.5, 47.5, theta1=-68.3, theta2=68.3, color=color, lw=lw, zorder=0))
        ax.plot([0, 14], [3, 3], color=color, lw=lw, zorder=0)
        ax.plot([0, 14], [47, 47], color=color, lw=lw, zorder=0)
        # Hoop & Backboard
        ax.add_patch(Rectangle((4, 22), 0.2, 6, color="#ec7607", lw=2, zorder=0))
        ax.add_patch(Circle((5.25, 25), 0.75, color="#ec7607", fill=False, lw=2, zorder=0))

    # 4. Right Side Features
    if half_court in [None, 'right']:
        ax.add_patch(Rectangle((75, 17), 19, 16, color=color, fill=False, lw=lw, zorder=0))
        # 3PT Arc (180 +/- 68.3)
        ax.add_patch(Arc((94-5.25, 25), 47.5, 47.5, theta1=111.7, theta2=248.3, color=color, lw=lw, zorder=0))
        ax.plot([80, 94], [3, 3], color=color, lw=lw, zorder=0)
        ax.plot([80, 94], [47, 47], color=color, lw=lw, zorder=0)
        # Hoop & Backboard
        ax.add_patch(Rectangle((90, 22), 0.2, 6, color="#ec7607", lw=2, zorder=0))
        ax.add_patch(Circle((94-5.25, 25), 0.75, color="#ec7607", fill=False, lw=2, zorder=0))

def draw_plotly_court(xref='x', yref='y', half_court=None):
    """
    Returns a list of high-quality NBA court lines for Plotly.
    Refactored to return a list of dicts rather than modifying a fig.
    """
    def ellipse_arc(x_center, y_center, a, b, start_angle, end_angle, N=100):
        t = np.linspace(start_angle, end_angle, N)
        x = x_center + a * np.cos(t)
        y = y_center + b * np.sin(t)
        path = f'M {x[0]},{y[0]}'
        for k in range(1, len(t)):
            path += f' L {x[k]},{y[k]}'
        return path

    line_col = "#777777"
    three_r = 23.75
    
    # Base shapes
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=94, y1=50, line=dict(color=line_col, width=2), layer='below'),
        dict(type="line", x0=47, y0=0, x1=47, y1=50, line=dict(color=line_col, width=2), layer='below')
    ]

    # Midcourt Circle
    if half_court is None:
        shapes.append(dict(type="circle", x0=41, y0=19, x1=53, y1=31, line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'left':
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, np.pi/2, 3*np.pi/2), line=dict(color=line_col, width=2), layer='below'))
    elif half_court == 'right':
        shapes.append(dict(type="path", path=ellipse_arc(47, 25, 6, 6, -np.pi/2, np.pi/2), line=dict(color=line_col, width=2), layer='below'))

    # Side Features
    if half_court in [None, 'left']:
        shapes += [
            dict(type="rect", x0=0, y0=17, x1=19, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=3, x1=14, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=0, y0=47, x1=14, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(5.25, 25, three_r, three_r, -1.18, 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=4, y0=22, x1=4.2, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=4.5, y0=24.25, x1=6, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]

    if half_court in [None, 'right']:
        shapes += [
            dict(type="rect", x0=75, y0=17, x1=94, y1=33, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=3, x1=94, y1=3, line=dict(color=line_col, width=1), layer='below'),
            dict(type="line", x0=80, y0=47, x1=94, y1=47, line=dict(color=line_col, width=1), layer='below'),
            dict(type="path", path=ellipse_arc(94-5.25, 25, three_r, three_r, np.pi - 1.18, np.pi + 1.18), line=dict(color=line_col, width=1), layer='below'),
            dict(type="rect", x0=94-4.2, y0=22, x1=94-4, y1=28, line=dict(color="#ec7607", width=2), fillcolor='#ec7607'),
            dict(type="circle", x0=94-6, y0=24.25, x1=94-4.5, y1=25.75, line=dict(color="#ec7607", width=2)),
        ]

    # Assign coordinate references
    for s in shapes:
        s['xref'] = xref
        s['yref'] = yref
    
    return shapes
