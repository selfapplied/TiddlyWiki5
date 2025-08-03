from sympy import symbols, Function, simplify, diff, expand, Eq, solveset, S
from sympy.abc import s, n
from collections import defaultdict

class SymbolicAttractor:
    def __init__(self, expression):
        self.expr = expression
        self.history = [expression]

    def apply_transform(self, transform):
        new_expr = transform(self.expr)
        self.history.append(new_expr)
        self.expr = new_expr

    def is_fixed_point(self):
        return len(self.history) >= 2 and self.history[-1] == self.history[-2]

    def __repr__(self):
        return f"Attractor({self.expr})"

class SymbolicEngine:
    def __init__(self):
        self.transforms = []
        self.attractors = []

    def add_transform(self, transform_func):
        self.transforms.append(transform_func)

    def seed(self, expr):
        attractor = SymbolicAttractor(expr)
        self.attractors.append(attractor)
        return attractor

    def run(self, max_iters=10):
        for attractor in self.attractors:
            for _ in range(max_iters):
                for transform in self.transforms:
                    attractor.apply_transform(transform)
                    if attractor.is_fixed_point():
                        break

    def get_fixed_points(self):
        return [a for a in self.attractors if a.is_fixed_point()] 