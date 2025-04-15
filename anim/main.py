from manim import *
import random
import numpy as np
import torch

# from manim_slides import Slide

class DottedLine(Line):

    """A dotted :class:`Line`.
    Parameters
    ----------
    args : Any
        Arguments to be passed to :class:`Line`
    dot_spacing : Optional[:class:`float`]
        Minimal spacing of the dots. The spacing is scaled up to fit the start and end of the line.
    dot_kwargs : Any
        Arguments to be passed to ::class::`Dot`
    kwargs : Any
        Additional arguments to be passed to :class:`Line`
    Examples
    --------
    .. manim:: DottedLineExample
        :save_last_frame:
        class DottedLineExample(Scene):
            def construct(self):
                # default dotted line
                dotted_1 = DottedLine(LEFT, RIGHT))
                # reduced spacing
                dotted_2 = DottedLine(LEFT, RIGHT, dot_spacing=.3).shift(.5*DOWN))
                # smaller and colored dots
                dotted_3 = DottedLine(LEFT, RIGHT, dot_kwargs=dict(radius=.04, color=YELLOW)).shift(DOWN))

                self.add(dotted_1, dotted_2, dotted_3)

    """

    def __init__(
        self,
        *args,
        dot_spacing=0.4,
        dot_kwargs={},
        **kwargs
    ):
        Line.__init__(self, *args, **kwargs)
        n_dots = int(self.get_length() / dot_spacing) + 1
        dot_spacing = self.get_length() / (n_dots - 1)
        unit_vector = self.get_unit_vector()
        start = self.start

        self.dot_points = [start + unit_vector * dot_spacing * x for x in range(n_dots)]
        self.dots = [Dot(point, **dot_kwargs) for point in self.dot_points]

        self.clear_points()

        self.add(*self.dots)

        self.get_start = lambda: self.dot_points[0]
        self.get_end = lambda: self.dot_points[-1]

    def get_first_handle(self):
        return self.dot_points[-1]

    def get_last_handle(self):
        return self.dot_points[-2]

class DirectedLine(Line):

    """A directional :class:`Line`.
    Parameters
    ----------
    args : Any
        Arguments to be passed to :class:`Line`
    triangle_spacing : Optional[:class:`float`]
        Minimal spacing of the triangles. The spacing is scaled up to fit the start and end of the line.
    triangle_kwargs : Any
        Arguments to be passed to ::class::`Triangle`
    kwargs : Any
        Additional arguments to be passed to :class:`Line`
    Examples
    --------
    .. manim:: DirectedLineExample
        :save_last_frame:
        class DirectedLineExample(Scene):
            def construct(self):
                # default dotted line
                dotted_1 = DottedLine(LEFT, RIGHT))
                # reduced spacing
                dotted_2 = DottedLine(LEFT, RIGHT, dot_spacing=.3).shift(.5*DOWN))
                # smaller and colored dots
                dotted_3 = DottedLine(LEFT, RIGHT, dot_kwargs=dict(radius=.04, color=YELLOW)).shift(DOWN))

                self.add(dotted_1, dotted_2, dotted_3)

    """

    def __init__(
        self,
        *args,
        spacing=0.4,
        triangle_kwargs={},
        **kwargs
    ):
        Line.__init__(self, *args, **kwargs)
        n_dots = int(self.get_length() / spacing) + 1
        dot_spacing = self.get_length() / (n_dots - 1)
        unit_vector = self.get_unit_vector()
        angle = -PI / 2 + angle_of_vector(unit_vector)
        start = self.start
        end = self.end

        self.dot_points = [start + unit_vector * dot_spacing * x for x in range(n_dots)]
        self.dots = [Triangle(fill_opacity=1, **triangle_kwargs).scale(0.15).rotate_about_origin(angle).shift(point) for point in self.dot_points]

        self.clear_points()

        self.add(*self.dots)
        self.add(Line(start, end, stroke_width=10.0))

        self.get_start = lambda: self.dot_points[0]
        self.get_end = lambda: self.dot_points[-1]

    def get_first_handle(self):
        return self.dot_points[-1]

    def get_last_handle(self):
        return self.dot_points[-2]

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

heatmap = color_gradient([BLUE, PURPLE, GRAY, BS381.AZO_ORANGE, GOLD], 13)
bwheatmap = color_gradient([DARK_GRAY, WHITE], 8)

def get_color(colors, alpha):
    i = int(clamp(alpha * len(colors), 0, len(colors) - 1))
    return colors[i]

def random_scalar():
    return random.normalvariate()

def random_color(colors=None):
    if colors is None:
        colors = heatmap
    alpha = random.random()
    return get_color(colors, alpha)

def make_square(v=None):
    x = random_scalar() if v is None else v
    alpha = clamp((x + 2.0) / 4.0, 0.0, 1.0)
    color = get_color(heatmap, alpha)
    vg = VGroup()
    square = Square(color=color, fill_opacity=1)
    text = Text(f"{x:.2f}")
    vg.add(square)
    vg.add(text)
    return vg

def vstack(items):
    g = VGroup(items).set_x(0).set_y(0).arrange(direction=DOWN)
    return g

def hstack(items):
    g = VGroup(items).set_x(0).set_y(0).arrange(buff=0.25)
    return g

class OpBox(VGroup):
    def __init__(self, txt):
        s = Square(fill_opacity=1.0, fill_color=BLACK)
        t = Text(txt, font_size=96.0)
        self.s = s
        super().__init__(s, t)
    def get_inputs(self):
        return [self.s.get_center(), self.s.get_center()]
    def get_outputs(self):
        return [self.s.get_center(), self.s.get_center()]

class TextBox(VGroup):
    def __init__(self, txt):
        t = Tex(txt, font_size=96.0)
        self.t = t
        super().__init__(t)
    def get_inputs(self):
        return [self.t.get_top() + UP * 0.2]
    def get_outputs(self):
        return [self.t.get_bottom() + DOWN * 0.2]

class Splitter(VGroup):
    def __init__(self):
        s = Circle(fill_opacity=1.0, fill_color=BLACK).scale(0.2)
        self.s = s
        super().__init__(s)
    def get_inputs(self):
        return [self.s.get_center(), self.s.get_center()]
    def get_outputs(self):
        return [self.s.get_center(), self.s.get_center()]

def op_box(txt):
    s = Square(fill_opacity=1.0, fill_color=BLACK)
    t = Text(txt, font_size=96.0)
    return VGroup(s, t)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Activation(VGroup):
    def __init__(self, horizontal=False, height=1.0, tanh=False):
        s = Rectangle(fill_opacity=1.0, width=2.0, height=2.0 * height, fill_color=BLACK)
        p1 = np.array([-1, -0.8 * height, 0])
        p2 = np.array([0, -0.8 * height, 0])
        p3 = np.array([0, 0.8 * height, 0])
        p4 = np.array([1, 0.8 * height, 0])
        b = CubicBezier(p1, p2, p3, p4, color=BLUE)
        a = Line(LEFT + DOWN * 0.8 * height, RIGHT + DOWN * 0.8 * height, stroke_width=2.0) if not tanh else Line(LEFT, RIGHT, stroke_width=2.0)
        if horizontal:
            self.inputs = [s.get_left()]
            self.outputs = [s.get_right()]
        else:
            self.inputs = [s.get_top()]
            self.outputs = [s.get_bottom()]
        super().__init__(s, a, b)

class LinearActivation(VGroup):
    def __init__(self, inputs=1, txt=r"$d_{in}, d_{out}$", activation_height=1.0, tanh=False):
        ab = Activation(height=activation_height, tanh=tanh).set_z_index(1)
        linear = VGroup(
            Rectangle(width=2.0, height=0.5, fill_opacity=1, fill_color=BLACK).set_z_index(1),
            Tex(txt).scale(0.8),
        ).set_z_index(1).next_to(ab, UP)
        c = Line(linear[0].get_bottom(), ab.get_top())
        cxs = VGroup([Line(ORIGIN, UP * 0.2) for i in range(inputs)]).arrange(buff=2.0 / inputs).shift(linear[0].get_top()).shift(UP * 0.1)
        cy = Line(ab.get_bottom(), ab.get_bottom() + DOWN * 0.2)
        self.cy = cy 
        self.cxs = cxs
        super().__init__(linear, ab, cxs, c, cy)
    def get_inputs(self):
        return [c.get_top() for c in self.cxs]
    def get_outputs(self):
        return [self.cy.get_center()]

class LinearScene(Scene):
    def construct(self):
        random.seed(124)

        self.play(Create(Text("Neural Network Layer").shift(UP * 3)))
        self.wait(1)

        fade = VGroup()
        f_in = 6
        f_out = 5

        g = VGroup([
            make_square() for i in range(f_in)
        ]).set_x(0).set_y(0).set_z_index(1).arrange(direction=DOWN, buff=1.5).scale(0.2)
        self.play(Create(g))
        self.play(g.animate.shift(LEFT * 2))
        fade.add(g)

        lbl_x = Tex('$x$')
        lbl_x.next_to(g, LEFT)
        self.play(Create(lbl_x))
        fade.add(lbl_x)

        hvals = [ random_scalar() for i in range(f_out) ]
        h = VGroup([
            make_square(hvals[i]) for i in range(f_out)
        ]).set_x(0).set_y(0).set_z_index(1).arrange(direction=DOWN, buff=1.5).scale(0.2).shift(RIGHT * 2)
        self.play(Create(h))
        fade.add(h)

        lines = VGroup()
        for gi in g:
            for hi in h:
                lines.add(Line(gi.get_center(), hi.get_center(), color=random_color(colors=bwheatmap)))
        self.play(Create(lines))
        fade.add(lines)

        act_boxes = VGroup([
            Activation() for i in range(f_out)
        ]).set_x(0).set_y(0).set_z_index(1).arrange(direction=DOWN, buff=1.5).scale(0.2).next_to(h, RIGHT)
        self.play(Create(act_boxes))
        fade.add(act_boxes)
        connectors = VGroup()
        for i in range(f_out):
            connectors.add(Line(h[i].get_center(), act_boxes[i].get_center()))
        self.play(Create(connectors))
        fade.add(connectors)

        h_act = VGroup([
            make_square(sigmoid(hvals[i])) for i in range(f_out)
        ]).set_x(0).set_y(0).set_z_index(1).arrange(direction=DOWN, buff=1.5).scale(0.2).next_to(act_boxes, RIGHT)
        self.play(Create(h_act))
        fade.add(h_act)

        connectors2 = VGroup()
        for i in range(f_out):
            connectors2.add(Line(h[i].get_center(), h_act[i].get_center()))
        self.play(Create(connectors2))
        fade.add(connectors2)

        lbl_y = Tex('$y$')
        lbl_y.next_to(h_act, RIGHT)
        self.play(Create(lbl_y))
        fade.add(lbl_y)
        self.wait(1)

        self.play(Create(Tex(r"$y=\sigma(Wx+b)$").shift(DOWN * 3)))
        self.wait(1)

        self.play(Create(Tex(r"$\sigma(x)=\frac{1}{1+e^{-x}}$").shift(DOWN * 3 + RIGHT * 4)))
        self.wait(1)

class LinearScene2(Scene):
    def construct(self):
        random.seed(124)
        self.play(Create(Text("Neural Network Layer").shift(UP * 3)))
        self.play(Create(Tex(r"$y=\sigma(Wx+b)$").shift(DOWN * 3)))
        self.wait(1)

        b = LinearActivation()
        lbl_x = Tex("$x$").next_to(b, UP)
        lbl_y = Tex("$y$").next_to(b, DOWN)
        v = VGroup(lbl_x, b, lbl_y)
        self.play(Create(v))
        self.wait(1)

class MultiInput(Scene):
    def construct(self):
        self.play(Create(Text("Multi-input Neural Network").shift(UP * 3)))
        self.wait(1)

        b = LinearActivation(txt=r"$d_{in}, d_{out}$", inputs=2)
        top_left_lbl = Tex("$x$").next_to(b.inputs[0], UP)
        top_right_lbl = Tex("$h$").next_to(b.inputs[1], UP)
        lbl_y = Tex("$y$").next_to(b, DOWN)
        v = VGroup(top_left_lbl, top_right_lbl, b, lbl_y).center()
        self.play(FadeIn(v))
        self.wait(1)

        self.play(Create(Tex(r"$y=\sigma(Wx+Uh+b)$").shift(DOWN * 3)))
        self.wait(1)

class MLP(Scene):
    def construct(self):
        random.seed(124)
        self.play(Create(Text("Multilayer Perceptron").shift(UP * 3)))
        self.wait(1)

        b = LinearActivation(txt=r'784, 200', activation_height=0.5)
        lbl_x = Tex("$x$").next_to(b, UP)
        c = LinearActivation(txt=r'200, 200', activation_height=0.5).next_to(b, DOWN, buff=-0.05)
        d = LinearActivation(txt=r'200, 10', activation_height=0.5).next_to(c, DOWN, buff=-0.05)
        lbl_y = Tex("$y$").next_to(d, DOWN)
        v = VGroup(lbl_x, b, c, d, lbl_y).move_to(ORIGIN).scale(0.8).shift(DOWN * 0.5)
        self.play(Create(v))
        self.wait(1)

        digit = ImageMobject("mnist3.png").scale(1.0).next_to(lbl_x, RIGHT).shift(RIGHT)
        digit.set_resampling_algorithm(RESAMPLING_ALGORITHMS['nearest'])
        self.add(digit)
        out = Text('0 0 0 1 0 0 0 0 0 0').scale(0.5).next_to(lbl_y, RIGHT).shift(RIGHT)
        self.add(out)
        self.wait(1)

def gru():
    v = VGroup()
    node_positions = [
        (-2, -1),   # 0
        (2, -1),    # 1
        (-1, -6),   # 2
        (-2, -3),   # 3
        (2, -8),    # 4
        (4, -7),    # 5
        (4, -3),    # 6
        (2, -10),   # 7
        (-1, 3),    # 8
        (0, 3),     # 9
        (3, 3),     # 10
        (-4, 2),    # 11
        (-3, 2),    # 12
        (1, 2),     # 13
        (6, 2),     # 14
        (-4, -3),   # 15
        (2, -3),    # 16
        (6, -7),    # 17
        (-1, -8),   # 18
        (4, -10),   # 19
        (2, -12),   # 20
        (-4, 5),    # 21
        (0, 5),     # 22
    ]
    node_contents = [
        LinearActivation(txt='Reset', inputs=2, activation_height=0.5),                 # 0
        LinearActivation(txt=r'$\hat{h}$', inputs=2, activation_height=0.5),            # 1
        LinearActivation(txt=r'Update', inputs=2, activation_height=0.5, tanh=True),    # 2
        OpBox('×').scale(0.5),                                                          # 3
        OpBox('×').scale(0.5),                                                          # 4
        OpBox('×').scale(0.5),                                                          # 5
        OpBox('1-').scale(0.5),                                                         # 6
        OpBox('+').scale(0.5),                                                          # 7
        Splitter(),                                                                     # 8
        Splitter(),                                                                     # 9
        Splitter(),                                                                     # 10
        Splitter(),                                                                     # 11
        Splitter(),                                                                     # 12
        Splitter(),                                                                     # 13
        Splitter(),                                                                     # 14
        Splitter(),                                                                     # 15
        Splitter(),                                                                     # 16
        Splitter(),                                                                     # 17
        Splitter(),                                                                     # 18
        Splitter(),                                                                     # 19
        TextBox(r'$h_t$'),                                                              # 20
        TextBox(r'$h_{t-1}$'),                                                          # 21
        TextBox(r'$x_t$'),                                                              # 22
    ]
    node_edges = [
        ((0, 0), (3, 0)),
        ((1, 0), (16, 0)),
        ((2, 0), (18, 0)),
        ((3, 0), (2, 0)),
        ((4, 0), (7, 0)),
        ((5, 0), (19, 0)),
        ((6, 0), (5, 0)),
        ((7, 0), (20, 0)),
        ((8, 0), (0, 1)),
        ((9, 0), (2, 1)),
        ((10, 0), (1, 1)),
        # ((0, 0), (3, 0)),
        # ((1, 0), (5, 0)),
        # ((1, 0), (6, 1)),
        # ((2, 0), (5, 1)),
        # ((3, 0), (2, 1)),
        # ((4, 0), (7, 0)),
        # ((5, 0), (7, 1)),
        # ((6, 0), (4, 0)),
    ]
    pos_scale = 0.8
    for i in range(len(node_positions)):
        x, y = node_positions[i]
        o = node_contents[i].shift((RIGHT * x + UP * y) * pos_scale).set_z_index(1)
        print(o.get_inputs(), o.get_outputs())
        v.add(o)
    for edge in node_edges:
        (i, i_n), (j, j_n) = edge
        start = node_contents[i].get_outputs()[i_n]
        end = node_contents[j].get_inputs()[j_n]
        line = Line(start, end, buff=0, stroke_width=4.0)
        v.add(line)
    v.scale(0.5)
    return v

class GRUScene(Scene):
    def construct(self):
        f = 6
        self.add(gru().shift(UP * 3))
        # lbl_h_t = Tex('$h_t$')
        # lbl_h_t.shift(LEFT * 6.5)
        # self.add(lbl_h_t)
        # lbl_h_t1 = Tex('$h_{t+1}$')
        # lbl_h_t1.shift(RIGHT * 6.2)
        # self.add(lbl_h_t1)
        # lbl_x_t = Tex('$x_t$')
        # lbl_x_t.shift(LEFT * 3).shift(DOWN * 3.3)
        # self.add(lbl_x_t)

        # x_t = vstack([make_square() for i in range(f)])
        # x_t.scale(0.2)
        # self.add(x_t)

class TimeSeries(Scene):
    def construct(self):
        n = 5
        f = 6
        fh = 3
        lbl_x_t = Tex('$x_t$')
        lbl_x_t.shift(LEFT * 6.5).shift(DOWN * 1.5)
        self.play(Create(lbl_x_t))
        lbl_h_t = Tex('$h_t$')
        lbl_h_t.shift(LEFT * 6.5).shift(UP * 2.0)

        # VGroup(items).set_x(0).set_y(0).arrange(buff=0.25)
        x = VGroup([vstack([make_square() for i in range(f)]) for j in range(n)]).center().scale(0.3).arrange(buff=1.2).shift(DOWN * 1.5)
        self.play(Create(x))
        self.play(Create(lbl_h_t))

        # #self.wait(2.0)
        # #self.play(x.animate.arrange(buff=0.5))
        # #self.wait(1.0)
        # x.arrange(buff=1.0).shift(DOWN * 1.5)
        h_ts = []
        def ms(i):
            return make_square(0.0) if i == 0 else make_square()
        h_ts = VGroup([vstack([ms(j) for i in range(fh)]) for j in range(n + 1)]).center().scale(0.3).arrange(buff=1.2).shift(UP * 2.0)

        # Op boxes
        def arrow_opbox():
            o = op_box("").scale(0.4)
            la = Arrow(start=LEFT * 1.0, end=RIGHT, buff=0, max_stroke_width_to_length_ratio=5).scale(0.4).next_to(o, LEFT, buff=0)
            ra = Arrow(start=LEFT, end=RIGHT * 1.0, buff=0, max_stroke_width_to_length_ratio=5).scale(0.4).next_to(o, RIGHT, buff=0)
            ba = Arrow(start=DOWN * 5.5, end=UP, buff=0, max_stroke_width_to_length_ratio=5/4).scale(0.4).next_to(o, DOWN, buff=0)
            return VGroup(la, ba, o, ra)

        boxes = VGroup([
            arrow_opbox().scale(0.5).next_to(x[i], UP, buff=0)
            for i in range(n)])

        for i in range(n):
            self.play(Create(h_ts[i]))
            if i > 0:
                self.play(Create(Arrow(start=DOWN, end=UP * 0.8).scale(0.8).next_to(h_ts[i], UP, buff=0)))
            self.play(Create(boxes[i]))
        self.play(Create(h_ts[n]))
        self.play(Create(Arrow(start=DOWN, end=UP * 0.8).scale(0.8).next_to(h_ts[n], UP, buff=0)))


def train():
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    global data_x
    global data_y
    global m_history
    global b_history
    global loss_history

    xpts = np.linspace(0, 20, 100) + np.random.normal(0.0, 1.0, 100)
    m_actual = np.random.normal(0.0, 1.0)
    b_actual = np.random.normal(0.0, 5.0)
    ypts = m_actual * xpts + b_actual + np.random.normal(0.0, 1.0, 100)

    # Defining the model architecture.
    class LinearRegressionModel(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.linear = torch.nn.Linear(1, 1)
            # this layer of the model has a single neuron, that takes in one scalar input and gives out one scalar output. 

        def forward(self, x): 
            y_pred = self.linear(x) 
            return y_pred 

    # Creating the model
    model = LinearRegressionModel()

    # Defining the Loss Function
    # Mean Squared Error is the most common choice of Loss Function for Linear Regression models.
    criterion = torch.nn.MSELoss()

    # Defining the Optimizer, which would update all the trainable parameters of the model, making the model learn the data distribution better and hence fit the distribution better.
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) 

    # We also need to convert all the data into tensors before we could use them for training our model.
    data_x = torch.tensor(xpts, dtype=torch.float)
    data_y = torch.tensor(ypts, dtype=torch.float)

    loss_history = []
    m_history = []
    b_history = []

    EPOCHS = 3500
    for epoch in range(EPOCHS):
        # We need to clear the gradients of the optimizer before running the back-propagation in PyTorch
        optimizer.zero_grad()
        # Feeding the input data in the model and getting out the predictions
        pred_y = model(data_x.reshape(-1, 1))
        # Calculating the loss using the model's predictions and the real y values
        loss = criterion(pred_y, data_y.reshape(-1, 1))
        # Back-Propagation
        loss.backward()
        # Updating all the trainable parameters
        optimizer.step()
        # Appending the loss.item() (a scalar value)
        loss_history.append(loss.item())
        # Appending the learnt slope and intercept   
        m_history.append(model.linear.weight.item())
        b_history.append(model.linear.bias.item())
        # We print out the losses after every 2000 epochs
        if (epoch) % 100 == 0:
            print('loss: ', loss.item(), model.linear.weight.item(), model.linear.bias.item())

class Loss(Scene):
    def construct(self):
        train()
        lh = loss_history[:1000]
        axes = Axes(
            x_range=[0, len(lh), 100],
            y_range=[-1, 3, 1],
            x_length=10,
            axis_config={'color': GREEN},
            tips=False,
            x_axis_config={
                'numbers_to_include': np.arange(0, len(lh)+1, 500),
            },
            y_axis_config={
                'scaling': LogBase(custom_labels=True),
                'include_numbers': True,
            },
        )
        axes_labels = axes.get_axis_labels(x_label='Epoch', y_label='Loss')
        graph = axes.plot_line_graph(range(len(lh)), lh, add_vertex_dots=False)
        vgraph = VGroup(axes, axes_labels)
        self.play(Create(vgraph))
        self.wait(1)
        self.play(Create(graph))
        self.wait(1)

class SGD(Scene):
    def construct(self):
        train()
        lbl = MathTex(r'{{ y }} = {{ m }} {{ x }} + {{ b }}')
        lbl.shift(UP * 3.3)
        lbl.set_color_by_tex("m", YELLOW)
        lbl.set_color_by_tex("b", YELLOW)

        axes = Axes(
            x_range=[0, 20, 1],
            y_range=[0, 10, 1],
            x_length=10,
            axis_config={'color': GREEN},
            x_axis_config={
                'numbers_to_include': np.arange(0, 21, 5),
                'numbers_with_elongated_ticks': np.arange(0, 21, 10),
            },
            tips=False,
        )
        axes_labels = axes.get_axis_labels()
        dots = VGroup(*[
            Dot(axes.c2p(x, y)) for x, y in zip(data_x, data_y)
        ])

        e = 150
        m_e, b_e = m_history[e], b_history[e]
        pi = 30
        x, y = data_x[pi], data_y[pi]
        ly = m_e * x + b_e
        vline = VGroup(
            Line(axes.c2p(x, y), axes.c2p(x, ly), stroke_width=2.0, color=YELLOW),
            Dot(axes.c2p(x, y), color=RED),
        )

        plot = VGroup(axes, axes_labels, dots)
        self.play(Create(plot))
        self.wait(1)
        self.add(plot)
        self.play(Create(lbl))
        self.add(lbl)
        self.wait(1)

        lines = VGroup()
        for i in range(e, 3500, 20):
            m_i, b_i = m_history[i], b_history[i]
            line = axes.plot(lambda x: m_i * x + b_i, color=BLUE)
            line.set_opacity(0.0)
            lines.add(line)
        lines[0].set_opacity(1.0)
        plot.add(lines)
        self.play(Create(lines[0]))
        self.wait(1)

        self.play(Create(vline))
        plot.add(vline)
        self.wait(1)
        self.play(plot.animate.scale(0.7))
        self.play(plot.animate.shift(UP * 0.5 + LEFT * 1.5))
        self.wait(1)

        loss = MathTex(r'L( m, b ) = \sum_i \left( m x_i + b - y_i \right)^2')
        loss.shift(DOWN * 3.0)
        for i in [2, 4, 10, 14]:
            loss[0][i].set_color(YELLOW)        
        self.add(loss)
        self.play(Create(loss))
        self.wait(1)

        update = MathTex(r'm & \leftarrow m - \alpha \frac{\partial L}{\partial m} \\ b & \leftarrow b - \alpha \frac{\partial L}{\partial b} ')
        for i in [0, 2, 9, 10, 12, 19]:
            update[0][i].set_color(YELLOW)
        for i in [4, 14]:
            update[0][i].set_color(TEAL)
        update.shift(RIGHT * 4.5)
        self.add(update)
        self.play(Create(update))
        self.wait(1)

        self.play(plot[-1].animate.set_opacity(0.0))
        self.wait(1)
        # 150 350 550 750 1000
        for i in [150, 350, 550, 750, 1000]:
            m_i, b_i = m_history[i], b_history[i]
            line = axes.plot(lambda x: m_i * x + b_i, color=BLUE)
            line.set_opacity(1.0)
            self.play(Transform(plot[-2][0], line))
        self.wait(1)
