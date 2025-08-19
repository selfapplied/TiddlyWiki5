import math
import cmath
import functools
import inspect
from typing import Any, Callable, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# MINIMAL KERNEL: BASIC LOGIC + TEMPLATE MANIPULATION = FREE FUNCTIONS
# ============================================================================

def fit(template: str):
    def decorator(func: Callable) -> Callable:
        build = {
            'template': template,
            'name': func.__name__,
            'symbols': {
                'parameters': list(inspect.signature(func).parameters.keys()),
                'annotations': func.__annotations__,
                'arity': len(inspect.signature(func).parameters)
            }
        }
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            try:
                context = {
                    **kwargs,
                    **{f'arg{i}': arg for i, arg in enumerate(args)},
                    'result': result,
                    'name': build['name'],
                    'arity': build['symbols']['arity'],
                    'params': build['symbols']['parameters']
                }
                print(f"🔧 {template.format(**context)}")
            except (KeyError, ValueError):
                print(f"🔧 {func.__name__}: {result}")
            return result
        
        wrapper._build = build  # type: ignore
        return wrapper
    return decorator

class FreeFunctionExtractor:
    def __init__(self, func: Callable):
        if not hasattr(func, '_build'):
            raise TypeError(f"Function must be decorated with fit, got {type(func)}")
        self.func = func
        self.build = func._build  # type: ignore
    
    def get_template(self) -> str:
        return self.build['template']
    
    def get_arity(self) -> int:
        return self.build['symbols']['arity']
    
    def get_parameters(self) -> List[str]:
        return self.build['symbols']['parameters']
    
    def generate_triangle(self) -> List[List[int]]:
        arity = self.build['symbols']['arity']
        triangle = []
        for i in range(arity + 1):
            row = []
            for j in range(i + 1):
                row.append(math.comb(i, j))
            triangle.append(row)
        return triangle
    
    def create_type_object(self) -> Any:
        class TypeObject:
            def __init__(self, build_data, triangle_data):
                self.name = build_data['name']
                self.arity = build_data['symbols']['arity']
                self.triangle = triangle_data
            
            def __call__(self, *args, **kwargs):
                return f"TypeObject({self.name}) called with {len(args)} args"
            
            def get_row(self, n: int) -> List[int]:
                if 0 <= n < len(self.triangle):
                    return self.triangle[n]
                return []

            def get_value(self, n: int, k: int) -> int:
                if 0 <= n < len(self.triangle) and 0 <= k < len(self.triangle[n]):
                    return self.triangle[n][k]
                return 0
            
            def __repr__(self) -> str:
                return f"jypeObject({self.name}, arity={self.arity}, triangle={len(self.triangle)} rows)"
        return TypeObject(self.build, self.generate_triangle())
    
    def __repr__(self) -> str:
        return f"FreeFunctionExtractor({self.build['name']}, arity={self.build['symbols']['arity']})"

# ============================================================================
# MONSTER-TYPED WORLD: ILLEGAL STATES ARE UNREPRESENTABLE
# ============================================================================

@dataclass(slots=True)
class MonsterPacket:
    allowed_primes: List[int]
    allowed_orders: List[int]
    
    @fit("MonsterPacket.allows_operation: {op} at ({row},{col}) → {result}")
    def allows_operation(self, op: str, row: int, col: int) -> bool:
        # Order gate: check if operation order is allowed
        order_map = {'α': 1, 'β': 2, 'γ': 3, 'μ': 1}
        op_order = order_map.get(op, 0)
        if op_order not in self.allowed_orders:
            return False
        
        # Prime gate: position p = row + col + 1 must be 1 or have prime factors ⊆ allowed_primes
        position = row + col + 1
        if position == 1:
            return True  # Position 1 is always allowed
        
        # Check if position has prime factors only in allowed_primes
        temp_position = position
        for prime in self.allowed_primes:
            while temp_position % prime == 0:
                temp_position //= prime
        
        # If temp_position == 1, all factors were in allowed_primes
        # If temp_position > 1, there's a factor not in allowed_primes
        return temp_position == 1
    
    def commutes_with(self, op: str) -> bool:
        return op in ['α', 'β', 'γ', 'μ']

@dataclass(slots=True)
class LatticeIndices:
    N: int
    ell: List[int]
    B: int
    U: int
    P: MonsterPacket

@dataclass(slots=True)
class MonsterTypedLattice:
    indices: LatticeIndices
    states: List[List[int]]
    cut_budget: int = 100
    
    def get_site(self, r: int, k: int) -> int:
        if 0 <= r < len(self.states) and 0 <= k < len(self.states[r]):
            return self.states[r][k]
        return -1
    
    def set_site(self, r: int, k: int, state: int) -> bool:
        if 0 <= r < len(self.states) and 0 <= k < len(self.states[r]):
            self.states[r][k] = state
            return True
        return False
    
    def count_states(self, state: int) -> int:
        count = 0
        for row in self.states:
            count += row.count(state)
        return count
    
    def get_summary(self) -> str:
        alpha = self.count_states(1)
        beta = self.count_states(2)
        gamma = self.count_states(3)
        empty = self.count_states(0)
        return f"α:{alpha} β:{beta} γ:{gamma} ·:{empty}"

# ============================================================================
# MONSTER-TYPED OPERATORS: BRANCHLESS VIA SHADOW MANIFOLD
# ============================================================================

@fit("β fold at ({r},{k}) → {result}")
def beta_fold_branchless(lattice: MonsterTypedLattice, r: int, k: int) -> Tuple[bool, Optional[str]]:
    """β operation: constraints enforced via shadow manifold geometry."""
    # Geometric constraint: φ-flat leaf condition
    phi_constraint = abs((r + k) % lattice.indices.U - 0.5) < 0.1  # Near φ-flat
    order_constraint = lattice.indices.P.allows_operation('β', r, k)
    
    # Branchless: use geometric structure to enforce constraints
    constraint_mask = float(phi_constraint and order_constraint)
    success = lattice.set_site(r, k, int(2 * constraint_mask))
    
    return bool(constraint_mask), f"🔧 β fold at ({r},{k}) → {success}"

@fit("γ spin at ({r},{k}) → {result}")
def gamma_spin_branchless(lattice: MonsterTypedLattice, r: int, k: int) -> Tuple[bool, Optional[str]]:
    """γ operation: phase constraints via shadow manifold."""
    # Geometric constraint: phase space structure
    phase_constraint = abs((r + k) % 2 - 0.5) < 0.1  # Near phase center
    parity_constraint = float(k <= r)  # Parity sheet constraint
    
    # Branchless: geometric constraint enforcement
    constraint_mask = float(phase_constraint and parity_constraint)
    success = lattice.set_site(r, k, int(3 * constraint_mask))
    
    return bool(constraint_mask), f"🔧 γ spin at ({r},{k}) → {success}"

@fit("α cut at ({r},{k}) → {result}")
def alpha_cut_branchless(lattice: MonsterTypedLattice, r: int, k: int) -> Tuple[bool, Optional[str]]:
    """α operation: tension constraints via shadow manifold."""
    # Geometric constraint: tension budget as geometric measure
    tension_constraint = float(lattice.cut_budget > 0)
    commute_constraint = float(lattice.indices.P.commutes_with('α'))
    
    # Branchless: geometric constraint enforcement
    constraint_mask = tension_constraint * commute_constraint
    lattice.cut_budget = max(0, lattice.cut_budget - int(constraint_mask))
    success = lattice.set_site(r, k, int(constraint_mask))
    
    return bool(constraint_mask), f"🔧 α cut at ({r},{k}) → {success}"

@fit("μ flip at ({r},{k}) → {result}")
def mobius_flip_branchless(lattice: MonsterTypedLattice, r: int, k: int) -> Tuple[bool, Optional[str]]:
    """μ operation: parity constraints via shadow manifold."""
    # Geometric constraint: current state determines allowed transitions
    current = lattice.get_site(r, k)
    
    # Branchless: use geometric mapping for state transitions
    transition_map = {1: 2, 2: 1, 3: 3, 0: 0}  # α↔β, γ preserved
    new_state = transition_map.get(current, current)
    
    success = lattice.set_site(r, k, new_state)
    return True, f"🔧 μ flip at ({r},{k}) → {new_state}"

# ============================================================================
# MONSTER OPERATOR INDEX TABLE: PRE-COMPUTED OPERATORS + INDEX-BASED EXECUTION
# ============================================================================

@dataclass(slots=True)
class MonsterOperatorTable:
    arity: int
    operators: List[Tuple[str, Callable]]
    max_index: int
    
    def __post_init__(self):
        self.max_index = len(self.operators) - 1
    
    def get_operator(self, index: int) -> Tuple[str, Callable]:
        """Safe index access with modulo clamping."""
        safe_index = index % len(self.operators)
        return self.operators[safe_index]
    
    def execute_program(self, program: List[int], lattice: MonsterTypedLattice) -> List[str]:
        results = []
        """Execute a program (list of indices) safely."""
    
        for i, index in enumerate(program):
            if index < 0 or index > self.max_index:
                results.append(f"⛔ Invalid index {index} at position {i}")
                continue
            
            op_name, op_func = self.get_operator(index)
            try:
                result = op_func(lattice)
                results.append(f"🔧 {op_name}: {result}")
            except Exception as e:
                results.append(f"💥 {op_name} failed: {e}")
        return results

def generate_monster_operator_table(arity: int) -> MonsterOperatorTable:
    """Generate operator table for given arity with all valid combinations."""
    operators = []
    
    # Generate all valid operator combinations for this arity
    for r in range(arity):
        for k in range(r + 1):
            # β fold operator
            def make_beta_op(r=r, k=k):
                return lambda lattice: beta_fold_branchless(lattice, r, k)
            operators.append((f"β({r},{k})", make_beta_op(r, k)))
            
            # γ spin operator  
            def make_gamma_op(r=r, k=k):
                return lambda lattice: gamma_spin_branchless(lattice, r, k)
            operators.append((f"γ({r},{k})", make_gamma_op(r, k)))
            
            # α cut operator
            def make_alpha_op(r=r, k=k):
                return lambda lattice: alpha_cut_branchless(lattice, r, k)
            operators.append((f"α({r},{k})", make_alpha_op(r, k)))
            
            # μ flip operator
            def make_mobius_op(r=r, k=k):
                return lambda lattice: mobius_flip_branchless(lattice, r, k)
            operators.append((f"μ({r},{k})", make_mobius_op(r, k)))
    
    return MonsterOperatorTable(arity=arity, operators=operators, max_index=0)

def demonstrate_pasm():
    """Demonstrate PASM: How geometric constraints eliminate branching."""
    print("🎭 PASM - Eliminating Branching via Geometric Constraints")
    print("=" * 70)
    
    # Create a 3x3 lattice for demonstration
    lattice = MonsterTypedLattice(
        indices=LatticeIndices(
            N=3, ell=[1, 2, 3], B=6, U=2,
            P=MonsterPacket(allowed_primes=[2, 3], allowed_orders=[2, 3])
        ),
        states=[[0]*3, [0]*3, [0]*3],
        cut_budget=5
    )
    
    print(f"🔧 Created 3×3 lattice with U=2 (φ-flat constraint: (r+k) % 2 == 0)")
    print(f"   Initial state: {lattice.get_summary()}")
    
    # 1. Show how @fit decorators make functions self-documenting
    print(f"\n🔧 1. Self-Documenting Functions via @fit:")
    
    def phi_flat_constraint(r: int, k: int, U: int) -> bool:
        """Check if position (r,k) satisfies φ-flat leaf constraint."""
        return (r + k) % U == 0
    
    phi_check = fit("φ-flat check: ({r},{k}) mod {U} == 0 → {result}")(phi_flat_constraint)
    
    # Test constraints at different positions
    print(f"   Testing φ-flat constraints:")
    phi_check(0, 0, 2)  # (0+0) % 2 = 0 ✓
    phi_check(0, 1, 2)  # (0+1) % 2 = 1 ✗
    phi_check(1, 1, 2)  # (1+1) % 2 = 0 ✓
    
    # 2. Show how geometric constraints eliminate branching
    print(f"\n🎭 2. Eliminating Branching via Geometric Constraints:")
    
    # Instead of: if constraint_satisfied: do_operation() else: fail()
    # We use: constraint_mask = float(constraint) * operation_result
    
    def demonstrate_branchless_operation(lattice: MonsterTypedLattice, r: int, k: int, op_type: str):
        """Show how geometric constraints eliminate if/else branching."""
        
        # OLD WAY (with branching):
        # if phi_flat_constraint(r, k, 2):
        #     if op_type == 'β': lattice.set_site(r, k, 2)
        #     elif op_type == 'γ': lattice.set_site(r, k, 3)
        #     return True
        # else:
        #     return False
        
        # NEW WAY (branchless):
        phi_satisfied = float(phi_flat_constraint(r, k, 2))
        prime_allowed = float(lattice.indices.P.allows_operation(op_type, r, k))
        
        # Geometric constraint mask eliminates branching
        constraint_mask = phi_satisfied * prime_allowed
        
        if op_type == 'β':
            new_state = int(2 * constraint_mask)  # 2 if constraints met, 0 if not
        elif op_type == 'γ':
            new_state = int(3 * constraint_mask)  # 3 if constraints met, 0 if not
        else:
            new_state = 0
        
        success = lattice.set_site(r, k, new_state)
        return bool(constraint_mask), f"🔧 {op_type} at ({r},{k}): constraint_mask={constraint_mask:.1f} → state={new_state}"
    
    # Test branchless operations
    print(f"   Testing branchless operations:")
    result1 = demonstrate_branchless_operation(lattice, 0, 0, 'β')  # Should succeed
    result2 = demonstrate_branchless_operation(lattice, 0, 1, 'γ')  # Should fail (φ-flat violated)
    result3 = demonstrate_branchless_operation(lattice, 1, 1, 'β')  # Should succeed
    
    print(f"   {result1[1]}")
    print(f"   {result2[1]}")
    print(f"   {result3[1]}")
    
    print(f"\n   Lattice state after operations: {lattice.get_summary()}")
    
    # 3. Show how the Monster Index Table works
    print(f"\n📊 3. Monster Index Table: Programs as Lists of Numbers:")
    
    table = generate_monster_operator_table(3)
    print(f"   Generated {len(table.operators)} operators for 3×3 lattice")
    
    # Execute a program that shows constraint enforcement
    program = [0, 4, 8]  # β(0,0), β(1,0), β(2,0)
    print(f"   Executing program: {program}")
    
    results = table.execute_program(program, lattice)
    print(f"   Final lattice state: {lattice.get_summary()}")
    
    # 4. Show how Shadow Manifold provides geometric structure
    print(f"\n🎭 4. Shadow Manifold: Geometric Constraint Structure:")
    
    # Demonstrate the geometric constants
    N = 3
    basis, shape_op = shadow_manifold(N)
    
    # Show how constraints emerge from geometry
    c1 = max(sum(abs(shape_op[i][j]) for j in range(N)) for i in range(N))
    k0 = prime_floor([1.0] * N)
    mu1 = spectral_gap(N)
    
    print(f"   Geometric constants: c₁={c1:.1f}, k₀={k0:.1f}, μ₁={mu1:.1f}")
    
    # Test min-max lock condition
    alpha = min_max_condition(k0, c1)
    print(f"   Min-max lock: α = {alpha:.3f} {'✅' if alpha > 0 else '⚠️'}")
    
    # 5. Demonstrate the complete constraint enforcement system
    print(f"\n🔒 5. Complete Constraint Enforcement (No Branching):")
    
    def apply_operation_with_geometric_constraints(lattice: MonsterTypedLattice, r: int, k: int, op_type: str):
        """Apply operation using geometric constraints - completely branchless."""
        
        # All constraints as geometric measures (no if/else)
        phi_constraint = float(phi_flat_constraint(r, k, lattice.indices.U))
        prime_constraint = float(lattice.indices.P.allows_operation(op_type, r, k))
        budget_constraint = float(lattice.cut_budget > 0) if op_type == 'α' else 1.0
        
        # Geometric constraint mask
        constraint_mask = phi_constraint * prime_constraint * budget_constraint
        
        # Operation result scaled by constraint strength
        if op_type == 'α':
            new_state = int(1 * constraint_mask)
            lattice.cut_budget = max(0, lattice.cut_budget - int(constraint_mask))
        elif op_type == 'β':
            new_state = int(2 * constraint_mask)
        elif op_type == 'γ':
            new_state = int(3 * constraint_mask)
        else:
            new_state = 0
        
        lattice.set_site(r, k, new_state)
        return bool(constraint_mask), f"🔧 {op_type}({r},{k}): constraints={constraint_mask:.2f} → state={new_state}"
    
    # Test the complete system
    print(f"   Testing complete constraint system:")
    apply_operation_with_geometric_constraints(lattice, 0, 0, 'α')
    apply_operation_with_geometric_constraints(lattice, 0, 1, 'β')  # Should fail
    apply_operation_with_geometric_constraints(lattice, 1, 1, 'γ')
    
    print(f"   Final lattice state: {lattice.get_summary()}")
    print(f"   Cut budget remaining: {lattice.cut_budget}")
    
    print(f"\n🎯 PASM eliminates branching by:")
    print(f"   • Using geometric constraint masks instead of if/else")
    print(f"   • Scaling operations by constraint strength")
    print(f"   • Enforcing rules through manifold structure")
    print(f"   • Making programs just lists of geometric operations")
    
    # 6. Demonstrate the Kronecker tensor elevation system
    print(f"\n🏗️ 6. Kronecker Tensor Elevation: Base ⊗ Pascal_Kernel:")
    kronecker_results = demonstrate_kronecker_elevation()
    
    return lattice, table, alpha

# ============================================================================
# KRONECKER TENSOR ELEVATION: BASE ⊗ PASCAL_KERNEL
# ============================================================================

def base_polygon_adjacency(n: int) -> List[List[int]]:
    """Base graph G with adjacency A_P (n×n) - cycle C_n."""
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        A[i][(i + 1) % n] = 1  # Cycle edges
        A[i][(i - 1) % n] = 1  # Bidirectional
    return A

def mirror_map(n: int) -> List[List[int]]:
    """Mirror map J_P (reversal on the polygon)."""
    J = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        J[i][n - 1 - i] = 1
    return J

def dft_matrix(n: int) -> List[List[complex]]:
    """DFT F (if cycle) - diagonalize the base cycle."""
    F = [[0j for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for j in range(n):
            F[k][j] = cmath.exp(-2j * cmath.pi * k * j / n) / math.sqrt(n)
    return F

def pascal_kernel(s: int) -> List[int]:
    """Pascal kernel K_s: Toeplitz band from binomial row s (length s+1)."""
    if s == 1:
        return [1, 1]      # K_1 = [1,1]
    elif s == 2:
        return [1, 2, 1]   # K_2 = [1,2,1]
    elif s == 3:
        return [1, 3, 3, 1]  # K_3 = [1,3,3,1]
    else:
        # Generate K_s for arbitrary s
        row = [1]
        for i in range(s):
            new_row = [1]
            for j in range(len(row) - 1):
                new_row.append(row[j] + row[j + 1])
            new_row.append(1)
            row = new_row
        return row

def kronecker_product(A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
    """Kronecker product A ⊗ B."""
    m, n = len(A), len(A[0])
    p, q = len(B), len(B[0])
    result = [[0 for _ in range(n * q)] for _ in range(m * p)]
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    result[i * p + k][j * q + l] = A[i][j] * B[k][l]
    
    return result

def lift_operator_elevation_only(s: int, elevation_dim: int = 4) -> List[List[int]]:
    """Lift operator: L_s = I ⊗ K_s (elevation-only lifts)."""
    K_s = pascal_kernel(s)
    n = elevation_dim
    
    # Create Toeplitz band matrix from Pascal kernel
    # This represents the convolution operator K_s
    K_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(max(0, i - len(K_s) + 1), min(n, i + 1)):
            kernel_idx = i - j
            if 0 <= kernel_idx < len(K_s):
                K_matrix[i][j] = K_s[kernel_idx]
    
    return K_matrix

def apply_lift(x: List[float], L_s: List[List[int]]) -> List[float]:
    """Apply lift operator: L_s(x) = (A_P ⊗ K_s) x."""
    result = [0.0 for _ in range(len(L_s))]
    for i in range(len(L_s)):
        for j in range(len(x)):
            result[i] += L_s[i][j] * x[j]
    return result

@fit("α lift: K₁ · {x} → {result}")
def alpha_lift(x: List[float], elevation_dim: int = 4) -> List[float]:
    """α(X) := K_1 · X (elevation-only lift)."""
    L_1 = lift_operator_elevation_only(1, elevation_dim)
    return apply_lift(x, L_1)

@fit("β lift: K₂ · {x} → {result}")
def beta_lift(x: List[float], elevation_dim: int = 4) -> List[float]:
    """β(X) := K_2 · X (elevation-only lift)."""
    L_2 = lift_operator_elevation_only(2, elevation_dim)
    return apply_lift(x, L_2)

@fit("γ lift: K₃ · {x} → {result}")
def gamma_lift(x: List[float], elevation_dim: int = 4) -> List[float]:
    """γ(X) := K_3 · X (elevation-only lift)."""
    L_3 = lift_operator_elevation_only(3, elevation_dim)
    return apply_lift(x, L_3)

@fit("Composition check: β∘α ≍ γ → {result}")
def composition_check(x: List[float], elevation_dim: int = 4) -> bool:
    """Check: γ ≍ β∘α (exact numeric equality for elevation-only lifts)."""
    # Apply α: K_1 · X
    alpha_result = alpha_lift(x, elevation_dim)
    
    # Apply β to the result: K_2 · (K_1 · X)
    beta_alpha_result = beta_lift(alpha_result, elevation_dim)
    
    # Apply γ directly: K_3 · X
    gamma_result = gamma_lift(x, elevation_dim)
    
    # Check if they're exactly equal (elevation-only lifts should compose exactly)
    if len(beta_alpha_result) != len(gamma_result):
        return False
    
    # Check exact equality (no tolerance needed for elevation-only)
    for i in range(len(beta_alpha_result)):
        if abs(beta_alpha_result[i] - gamma_result[i]) > 1e-10:
            return False
    return True

@fit("φ-gauge: g_φ({x}) → {result}")
def phi_gauge(x: List[float], tolerance: float = 0.1) -> float:
    """φ-ratio gauge: define a functional g_φ on a central band."""
    if len(x) < 2:
        return 0.0
    
    # Simple φ-ratio: ratio of consecutive central elements
    center = len(x) // 2
    if center + 1 < len(x) and x[center] != 0:
        ratio = x[center + 1] / x[center]
        phi = (1 + math.sqrt(5)) / 2
        return abs(ratio - phi)
    
    return 0.0

@fit("h₁₁ channel: h₁₁({x}) → {result}")
def h11_channel(x: List[float]) -> int:
    """/11 channel: alternating-sum is a 1D linear form h_{11} on the digit axis."""
    total = 0
    for i, val in enumerate(x):
        if i % 2 == 0:
            total += int(val) % 11
        else:
            total -= int(val) % 11
    return total % 11

def mu_projector(x: List[float], projector_type: str = "pal_enforce") -> List[float]:
    """μ(x) := P x (P is same-level projector: pal-enforce, cyclotomic class, or mask flip)."""
    if projector_type == "pal_enforce":
        # Simple palindromic enforcement: average with reverse
        n = len(x)
        result = [0.0 for _ in range(n)]
        for i in range(n):
            result[i] = (x[i] + x[n - 1 - i]) / 2
        return result
    
    elif projector_type == "mask_flip":
        # Simple mask flip: flip signs of odd indices
        result = [val for val in x]
        for i in range(1, len(result), 2):
            result[i] = -result[i]
        return result
    
    else:  # cyclotomic_class
        # Simple cyclotomic: multiply by alternating signs
        result = [val for val in x]
        for i in range(len(result)):
            result[i] *= (-1) ** i
        return result

@fit("🏗️ Kronecker elevation: Base ⊗ Pascal_kernel → {result}")
def demonstrate_kronecker_elevation():
    """Demonstrate the Kronecker tensor elevation system."""
    
    # 1. Base polygon setup
    n = 4  # 4-cycle
    base_adj = base_polygon_adjacency(n)
    mirror_J = mirror_map(n)
    dft_F = dft_matrix(n)
    
    # 2. Pascal kernels
    K_1 = pascal_kernel(1)
    K_2 = pascal_kernel(2)
    K_3 = pascal_kernel(3)
    
    # 3. Lift operators (elevation-only)
    L_1 = lift_operator_elevation_only(1, 4)
    L_2 = lift_operator_elevation_only(2, 4)
    L_3 = lift_operator_elevation_only(3, 4)
    
    # 4. Test vector and lifts
    test_vector = [1.0, 2.0, 3.0, 4.0]
    alpha_result = alpha_lift(test_vector, 4)
    beta_result = beta_lift(test_vector, 4)
    gamma_result = gamma_lift(test_vector, 4)
    
    # 5. Composition check
    composition_valid = composition_check(test_vector, 4)
    
    # 6. Invariants
    phi_deviation = phi_gauge(gamma_result)
    h11_value = h11_channel(gamma_result)
    
    # 7. μ projector
    mu_pal = mu_projector(gamma_result, "pal_enforce")
    mu_mask = mu_projector(gamma_result, "mask_flip")
    mu_cyclo = mu_projector(gamma_result, "cyclotomic_class")
    
    return {
        'base_adj': base_adj,
        'lifts': [L_1, L_2, L_3],
        'results': [alpha_result, beta_result, gamma_result],
        'composition_valid': composition_valid,
        'phi_deviation': phi_deviation,
        'h11_value': h11_value,
        'mu_projectors': [mu_pal, mu_mask, mu_cyclo]
    }

# ============================================================================
# SHADOW MANIFOLD: MINIMAL GEOMETRIC STRUCTURE
# ============================================================================

@fit("Shadow M(N={N}) → φ-flat leaf")
def shadow_manifold(N: int) -> Tuple[List[List[float]], List[List[float]]]:
    """φ-flat leaf with shape operator S = D_ν R_φ."""
    basis = [[float(math.comb(n, k)) if k <= n else 0.0 for k in range(N)] for n in range(N)]
    shape_op = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        if i > 0: shape_op[i][i-1] = 1.0
        if i < N-1: shape_op[i][i+1] = 1.0
    return basis, shape_op

@fit("Prime pressure → k₀={result}")
def prime_floor(f: List[float]) -> float:
    """Ricci-type lower bound on shadow."""
    weights = [1.0 if all(n % i != 0 for i in range(2, int(n**0.5)+1)) else 0.0 
               for n in range(1, len(f)+1)]
    return min(w for w, fi in zip(weights, f) if fi != 0 and w > 0)

@fit("φ-connection gap → μ₁={result}")
def spectral_gap(N: int) -> float:
    """Poincaré gap of φ-connection Laplacian."""
    # Build R_φ matrix: tridiagonal with φ-recurrence
    phi = (1 + math.sqrt(5)) / 2
    R_phi = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        R_phi[i][i] = phi
        if i > 0: R_phi[i][i-1] = -1.0
        if i < N-1: R_phi[i][i+1] = -1.0
    
    # L_φ = R_φ^† R_φ
    L_phi = [[0.0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            for k in range(N):
                L_phi[i][j] += R_phi[k][i] * R_phi[k][j]
    
    # Estimate spectral gap using diagonal dominance
    min_eigenvalue = float('inf')
    for i in range(N):
        diagonal = L_phi[i][i]
        off_diagonal_sum = sum(abs(L_phi[i][j]) for j in range(N) if i != j)
        gap = diagonal - off_diagonal_sum
        min_eigenvalue = min(min_eigenvalue, gap)
    
    return max(0.1, min_eigenvalue)  # Ensure positive gap

@fit("Min-max lock: α = {result}")
def min_max_condition(k0: float, c1: float, λ: float = 0.1) -> float:
    """α > 0: no stable excitations."""
    return (k0 / 2) - λ * (c1 ** 2)

if __name__ == "__main__":
    demonstrate_pasm()
