from sympy import symbols, zeta, simplify, expand, sin, cos, pi, diff, exp, I, log, sqrt
from sympy.abc import s, n, p, k
from riemann_attractor import RiemannEngine, rotation_generator, scaling_generator, translation_generator
from collections import defaultdict

class MathematicalAttractorExplorer:
    def __init__(self):
        self.engine = RiemannEngine()
        self.prime_attractors = []
        self.geometric_attractors = []
        self.analytic_attractors = []
        
    def setup_prime_attractors(self):
        """Explore attractors related to prime number theory"""
        s = symbols('s')
        
        # Zeta function and its variants
        self.engine.seed(zeta(s), "riemann_zeta")
        self.engine.seed(zeta(s) * (1 - 2**(1-s)), "dirichlet_eta")
        self.engine.seed(zeta(s) * (1 - 3**(1-s)), "modified_zeta")
        
        # Prime counting functions
        self.engine.seed(log(zeta(s)), "log_zeta")
        self.engine.seed(diff(zeta(s), s), "zeta_derivative")
        
        # Add prime-specific transforms
        self.engine.add_transform(lambda e: e.subs(s, s + 1))
        self.engine.add_transform(lambda e: e.subs(s, 1/s))
        
    def setup_geometric_attractors(self):
        """Explore geometric and topological attractors"""
        x, y, z = symbols('x y z')
        
        # Geometric structures
        self.engine.seed(x**2 + y**2 + z**2, "sphere")
        self.engine.seed(x**2 + y**2 - z**2, "hyperboloid")
        self.engine.seed(x*y*z, "cubic_surface")
        self.engine.seed(exp(-(x**2 + y**2)), "gaussian")
        
        # Curvature-related functions
        self.engine.seed(1/(1 + x**2 + y**2), "curvature_function")
        self.engine.seed(sin(x)*cos(y), "wave_pattern")
        
        # Add geometric transforms
        self.engine.add_transform(lambda e: e.subs(x, x + y))
        self.engine.add_transform(lambda e: e.subs(y, y + z))
        
    def setup_analytic_attractors(self):
        """Explore analytic functions and their attractors"""
        z = symbols('z')
        
        # Complex analytic functions
        self.engine.seed(exp(z), "exponential")
        self.engine.seed(sin(z), "sine")
        self.engine.seed(cos(z), "cosine")
        self.engine.seed(log(z), "logarithm")
        self.engine.seed(z**2, "quadratic")
        self.engine.seed(1/z, "reciprocal")
        
        # Add analytic transforms
        self.engine.add_transform(lambda e: e.subs(z, z + 1))
        self.engine.add_transform(lambda e: e.subs(z, I*z))
        
    def explore_prime_distribution(self):
        """Analyze prime distribution patterns"""
        s = symbols('s')
        
        # Prime density functions
        self.engine.seed(zeta(s) / zeta(2*s), "prime_density")
        self.engine.seed(log(zeta(s)) / s, "logarithmic_density")
        
        # Möbius function related
        self.engine.seed(1 / zeta(s), "mobius_transform")
        
    def find_critical_points(self):
        """Find critical points where curvature or derivatives vanish"""
        critical_points = []
        
        for attractor in self.engine.attractors:
            # Find where first derivative vanishes
            try:
                if hasattr(attractor.expr, 'free_symbols'):
                    vars = list(attractor.expr.free_symbols)
                    if vars:
                        var = vars[0]
                        derivative = diff(attractor.expr, var)
                        if derivative == 0:
                            critical_points.append((attractor.name, "first_derivative_zero"))
            except:
                continue
                
        return critical_points
    
    def analyze_convergence(self):
        """Analyze convergence patterns of attractors"""
        convergence_report = {}
        
        for attractor in self.engine.attractors:
            if len(attractor.history) > 1:
                # Check if expression converges to a constant
                final_expr = attractor.history[-1]
                if not hasattr(final_expr, 'free_symbols') or len(final_expr.free_symbols) == 0:
                    convergence_report[attractor.name] = "converges_to_constant"
                elif len(attractor.history) >= 3:
                    # Check if oscillating
                    last_three = attractor.history[-3:]
                    if last_three[0] == last_three[2] and last_three[0] != last_three[1]:
                        convergence_report[attractor.name] = "oscillating"
                    else:
                        convergence_report[attractor.name] = "fixed_point"
                        
        return convergence_report
    
    def run_comprehensive_analysis(self):
        """Run full analysis of all mathematical attractors"""
        print("Setting up mathematical attractors...")
        self.setup_prime_attractors()
        self.setup_geometric_attractors()
        self.setup_analytic_attractors()
        self.explore_prime_distribution()
        
        # Add standard transforms
        self.engine.add_transform(simplify)
        self.engine.add_transform(expand)
        
        print("Running attractor evolution...")
        self.engine.run(max_iters=8)
        
        # Generate comprehensive report
        print("\n" + "="*50)
        print("MATHEMATICAL ATTRACTOR ANALYSIS")
        print("="*50)
        
        print("\n--- FIXED POINTS ---")
        for attractor in self.engine.get_fixed_points():
            print(f"{attractor.name}: {attractor.expr}")
            
        print("\n--- CRITICAL POINTS ---")
        critical = self.find_critical_points()
        for name, point_type in critical:
            print(f"{name}: {point_type}")
            
        print("\n--- CONVERGENCE ANALYSIS ---")
        convergence = self.analyze_convergence()
        for name, behavior in convergence.items():
            print(f"{name}: {behavior}")
            
        print("\n--- ZERO CURVATURE POINTS ---")
        zero_curv = self.engine.get_zero_curvature_points()
        for name, expr in zero_curv[:10]:  # Limit output
            print(f"{name}: {expr}")
            
        print("\n--- SYMMETRY ANALYSIS ---")
        symmetries = self.engine.analyze_symmetries()
        for symmetry_type, attractors in symmetries.items():
            print(f"{symmetry_type}: {attractors}")
            
        return {
            'fixed_points': self.engine.get_fixed_points(),
            'critical_points': critical,
            'convergence': convergence,
            'zero_curvature': zero_curv,
            'symmetries': symmetries
        }

# Specialized attractor for prime number patterns
class PrimeAttractor:
    def __init__(self, prime_function, name="prime_attractor"):
        self.function = prime_function
        self.name = name
        self.prime_values = []
        self.analytic_continuation = None
        
    def evaluate_at_primes(self, max_prime=20):
        """Evaluate function at prime numbers"""
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for p in primes:
            if p <= max_prime:
                try:
                    value = self.function.subs(symbols('s'), p)
                    self.prime_values.append((p, value))
                except:
                    continue
                    
    def find_analytic_continuation(self):
        """Find analytic continuation beyond natural domain"""
        # This would implement more sophisticated continuation logic
        pass 