import cmath


def make_differentiable_fn(name, fn, partial_derivative_fns):
    class Differentiable:
        def __init__(self, *xs):
            self.name = name
            assert(len(xs) == len(partial_derivative_fns))
            self.xs = xs
            self.x_vals = None

        def forward(self, values):  # Evaluate and store results in forward pass.
            self.x_vals = [x.forward(values) for x in self.xs]
            return fn(*self.x_vals)

        def backward(self, multiplier, derivatives):  # Use multivariate chain rule for backward pass.
            assert(self.x_vals is not None)
            for partial_derivative_fn, x in zip(partial_derivative_fns, self.xs):
                x.backward(multiplier * partial_derivative_fn(*self.x_vals), derivatives)

        def __add__(self, other):
            return add(self, other)

        def __sub__(self, other):
            return sub(self, other)

        def __mul__(self, other):
            return mul(self, other)

        def __truediv__(self, other):
            return div(self, other)

        def __pow__(self, other):
            return dpow(self, other)

    return Differentiable


sin = make_differentiable_fn('sin', cmath.sin, [cmath.cos])
cos = make_differentiable_fn('cos', cmath.cos, [lambda x: -cmath.sin(x)])
exp = make_differentiable_fn('exp', cmath.exp, [cmath.exp])
log = make_differentiable_fn('log', lambda x: cmath.log(x), [lambda x: 1/x])
add = make_differentiable_fn('+', lambda x, y: x + y, [lambda x, y: 1, lambda x, y: 1])
sub = make_differentiable_fn('-', lambda x, y: x - y, [lambda x, y: 1, lambda x, y: -1])
mul = make_differentiable_fn('*', lambda x, y: x*y, [lambda x, y: y, lambda x, y: x])
div = make_differentiable_fn('/', lambda x, y: x/y, [lambda x, y: 1/y, lambda x, y: -x/(y**2)])
dpow = make_differentiable_fn('pow', pow, [lambda x, y: y*pow(x, y-1), lambda x, y: cmath.log(x)*pow(x, y)])


def add_operators_to_class(c):
    for opname, fn in [('__add__', add), ('__sub__', sub), ('__mul__', mul), ('__truediv__', div), ('__pow__', dpow)]:
        def op(self, other, fn=fn):
            return fn(self, other)
        setattr(c, opname, op)


class DifferentiableVar:
    def __init__(self, name):
        self.name = name

    def forward(self, values):
        return values[self.name]

    def backward(self, multiplier, derivatives):
        derivatives[self.name] = derivatives.get(self.name, 0) + multiplier


class Const:
    def __init__(self, value):
        self.value = value

    def forward(self, values):
        return self.value

    def backward(self, multiplier, derivatives):
        pass


add_operators_to_class(DifferentiableVar)
add_operators_to_class(Const)


def inspect(f, values):
    result = f.forward(values)
    derivatives = {}
    f.backward(1, derivatives)
    return {'f': result, 'derivatives': derivatives}


def test():
    assert(inspect(Const(5), {}) == {'f': 5, 'derivatives': {}})
    x, y = [DifferentiableVar(name) for name in ['x', 'y']]
    assert(inspect(x**Const(10), {'x': -1}) == {'f': 1, 'derivatives': {'x': -10}})
    assert(inspect(x*y**Const(2) + Const(5)*x**Const(2)*y, {'x': -1, 'y': 2})
           == {'f': 6, 'derivatives': {'x': -16, 'y': 1}})
    assert(inspect(Const(3)*cos(y), {'y': 0}) == {'f': 3, 'derivatives': {'y': 0}})
    e = Const(cmath.exp(1).real)
    assert(inspect(e**y, {'y': 0}) == {'f': 1, 'derivatives': {'y': 1}})


if __name__ == '__main__':
    test()
