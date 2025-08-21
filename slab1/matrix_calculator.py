import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import warnings
warnings.filterwarnings('ignore')

class MatrixCalculator:
    def __init__(self):
        self.history = []
        self.matrices = {}  # Store named matrices
        
    def create_matrix(self, rows: int, cols: int, values: List[List[float]] = None, name: str = None) -> np.ndarray:
        """Create a matrix with given dimensions and values"""
        if values:
            matrix = np.array(values, dtype=float)
            if matrix.shape != (rows, cols):
                raise ValueError(f"Values don't match dimensions {rows}x{cols}")
        else:
            matrix = np.zeros((rows, cols))
        
        if name:
            self.matrices[name] = matrix
            
        return matrix
    
    def input_matrix_interactive(self, name: str = None) -> np.ndarray:
        """Interactive matrix input"""
        print(f"\n{'='*50}")
        print("MATRIX INPUT")
        print("="*50)
        
        try:
            rows = int(input("Enter number of rows: "))
            cols = int(input("Enter number of columns: "))
            
            print(f"\nEnter matrix elements ({rows}x{cols}):")
            print("You can enter:")
            print("1. Row by row (space-separated values)")
            print("2. Type 'random' for random matrix")
            print("3. Type 'identity' for identity matrix (square only)")
            print("4. Type 'zeros' for zero matrix")
            print("5. Type 'ones' for matrix of ones")
            
            choice = input("\nYour choice: ").strip().lower()
            
            if choice == 'random':
                matrix = np.random.randint(1, 10, (rows, cols)).astype(float)
            elif choice == 'identity':
                if rows != cols:
                    print("Identity matrix must be square! Creating square matrix...")
                    rows = cols = max(rows, cols)
                matrix = np.eye(rows)
            elif choice == 'zeros':
                matrix = np.zeros((rows, cols))
            elif choice == 'ones':
                matrix = np.ones((rows, cols))
            else:
                matrix = np.zeros((rows, cols))
                print(f"\nEnter values row by row:")
                for i in range(rows):
                    row_input = input(f"Row {i+1}: ")
                    row_values = [float(x) for x in row_input.split()]
                    if len(row_values) != cols:
                        raise ValueError(f"Row {i+1} must have {cols} values")
                    matrix[i] = row_values
            
            if name:
                self.matrices[name] = matrix
                print(f"✓ Matrix '{name}' created and stored")
            
            print(f"\nCreated matrix:")
            self.display_matrix(matrix)
            
            return matrix
            
        except ValueError as e:
            print(f"Error: {e}")
            return None
    
    def display_matrix(self, matrix: np.ndarray, title: str = "Matrix"):
        """Display matrix in a formatted way"""
        print(f"\n{title}:")
        print("-" * (len(title) + 1))
        
        if matrix.size == 1:
            print(f"[{matrix.item():.3f}]")
        else:
            # Format for better display
            formatted_matrix = np.array2string(matrix, 
                                             formatter={'float_kind': lambda x: f"{x:8.3f}"},
                                             separator='  ')
            print(formatted_matrix)
        
        print(f"Shape: {matrix.shape}")
        if matrix.ndim == 2:
            print(f"Determinant: {np.linalg.det(matrix):.3f}" if matrix.shape[0] == matrix.shape[1] else "Determinant: N/A (not square)")
    
    def add_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Add two matrices"""
        if A.shape != B.shape:
            raise ValueError(f"Matrices must have same shape: {A.shape} != {B.shape}")
        
        result = A + B
        self.history.append(f"Addition: {A.shape} + {B.shape} = {result.shape}")
        return result
    
    def subtract_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Subtract two matrices"""
        if A.shape != B.shape:
            raise ValueError(f"Matrices must have same shape: {A.shape} != {B.shape}")
        
        result = A - B
        self.history.append(f"Subtraction: {A.shape} - {B.shape} = {result.shape}")
        return result
    
    def multiply_matrices(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Multiply two matrices"""
        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Cannot multiply: {A.shape} × {B.shape} (inner dimensions don't match)")
        
        result = np.dot(A, B)
        self.history.append(f"Multiplication: {A.shape} × {B.shape} = {result.shape}")
        return result
    
    def scalar_multiply(self, matrix: np.ndarray, scalar: float) -> np.ndarray:
        """Multiply matrix by scalar"""
        result = matrix * scalar
        self.history.append(f"Scalar multiplication: {scalar} × {matrix.shape}")
        return result
    
    def transpose_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Transpose a matrix"""
        result = matrix.T
        self.history.append(f"Transpose: {matrix.shape} → {result.shape}")
        return result
    
    def determinant(self, matrix: np.ndarray) -> float:
        """Calculate determinant of a square matrix"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Determinant can only be calculated for square matrices")
        
        det = np.linalg.det(matrix)
        self.history.append(f"Determinant calculated for {matrix.shape} matrix")
        return det
    
    def inverse_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calculate matrix inverse"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Inverse can only be calculated for square matrices")
        
        det = np.linalg.det(matrix)
        if abs(det) < 1e-10:
            raise ValueError("Matrix is singular (determinant ≈ 0), inverse doesn't exist")
        
        result = np.linalg.inv(matrix)
        self.history.append(f"Inverse calculated for {matrix.shape} matrix")
        return result
    
    def eigenvalues_eigenvectors(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate eigenvalues and eigenvectors"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Eigenvalues can only be calculated for square matrices")
        
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        self.history.append(f"Eigenvalues/eigenvectors calculated for {matrix.shape} matrix")
        return eigenvalues, eigenvectors
    
    def rank_matrix(self, matrix: np.ndarray) -> int:
        """Calculate matrix rank"""
        rank = np.linalg.matrix_rank(matrix)
        self.history.append(f"Rank calculated for {matrix.shape} matrix")
        return rank
    
    def trace_matrix(self, matrix: np.ndarray) -> float:
        """Calculate matrix trace (sum of diagonal elements)"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Trace can only be calculated for square matrices")
        
        trace = np.trace(matrix)
        self.history.append(f"Trace calculated for {matrix.shape} matrix")
        return trace
    
    def power_matrix(self, matrix: np.ndarray, power: int) -> np.ndarray:
        """Raise matrix to a power"""
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix power can only be calculated for square matrices")
        
        result = np.linalg.matrix_power(matrix, power)
        self.history.append(f"Matrix power: {matrix.shape}^{power}")
        return result
    
    def solve_linear_system(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve linear system Ax = b"""
        if A.shape[0] != A.shape[1]:
            raise ValueError("Coefficient matrix must be square")
        if A.shape[0] != b.shape[0]:
            raise ValueError("Incompatible dimensions for A and b")
        
        try:
            solution = np.linalg.solve(A, b)
            self.history.append(f"Linear system solved: {A.shape} × x = {b.shape}")
            return solution
        except np.linalg.LinAlgError:
            raise ValueError("System has no unique solution (singular matrix)")
    
    def lu_decomposition(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """LU decomposition using scipy if available, else manual implementation"""
        try:
            from scipy.linalg import lu
            P, L, U = lu(matrix)
            self.history.append(f"LU decomposition for {matrix.shape} matrix")
            return L, U
        except ImportError:
            # Simple manual LU for demonstration
            n = matrix.shape[0]
            L = np.eye(n)
            U = matrix.copy()
            
            for i in range(n-1):
                for j in range(i+1, n):
                    if U[i, i] != 0:
                        factor = U[j, i] / U[i, i]
                        L[j, i] = factor
                        U[j, :] -= factor * U[i, :]
            
            return L, U
    
    def show_history(self):
        """Show operation history"""
        print(f"\n{'='*50}")
        print("OPERATION HISTORY")
        print("="*50)
        
        if not self.history:
            print("No operations performed yet.")
        else:
            for i, operation in enumerate(self.history, 1):
                print(f"{i:2}. {operation}")
    
    def show_stored_matrices(self):
        """Show all stored matrices"""
        print(f"\n{'='*50}")
        print("STORED MATRICES")
        print("="*50)
        
        if not self.matrices:
            print("No matrices stored.")
        else:
            for name, matrix in self.matrices.items():
                print(f"\n{name}: {matrix.shape}")
                print(matrix)
    
    def clear_history(self):
        """Clear operation history"""
        self.history.clear()
        print("✓ History cleared")
    
    def clear_matrices(self):
        """Clear stored matrices"""
        self.matrices.clear()
        print("✓ Stored matrices cleared")

def interactive_menu():
    """Interactive menu for matrix operations"""
    calc = MatrixCalculator()
    
    print("🔢 MATRIX OPERATIONS TOOL")
    print("="*60)
    print("Welcome to the Interactive Matrix Calculator!")
    
    while True:
        print(f"\n{'='*60}")
        print("MAIN MENU")
        print("="*60)
        print("1.  Create/Input Matrix")
        print("2.  Matrix Addition")
        print("3.  Matrix Subtraction") 
        print("4.  Matrix Multiplication")
        print("5.  Scalar Multiplication")
        print("6.  Matrix Transpose")
        print("7.  Calculate Determinant")
        print("8.  Matrix Inverse")
        print("9.  Eigenvalues & Eigenvectors")
        print("10. Matrix Rank")
        print("11. Matrix Trace")
        print("12. Matrix Power")
        print("13. Solve Linear System")
        print("14. LU Decomposition")
        print("15. Show Operation History")
        print("16. Show Stored Matrices")
        print("17. Clear History")
        print("18. Clear Stored Matrices")
        print("19. Exit")
        
        try:
            choice = input("\nEnter your choice (1-19): ").strip()
            
            if choice == '1':
                name = input("Enter matrix name (or press Enter to skip): ").strip()
                name = name if name else None
                calc.input_matrix_interactive(name)
                
            elif choice == '2':
                print("\nMatrix Addition: A + B")
                if len(calc.matrices) >= 2:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name1 = input("Enter first matrix name: ").strip()
                    name2 = input("Enter second matrix name: ").strip()
                    
                    if name1 in calc.matrices and name2 in calc.matrices:
                        result = calc.add_matrices(calc.matrices[name1], calc.matrices[name2])
                        calc.display_matrix(result, f"{name1} + {name2}")
                    else:
                        print("One or both matrices not found!")
                else:
                    print("Need at least 2 stored matrices. Create matrices first.")
                    
            elif choice == '3':
                print("\nMatrix Subtraction: A - B")
                if len(calc.matrices) >= 2:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name1 = input("Enter first matrix name: ").strip()
                    name2 = input("Enter second matrix name: ").strip()
                    
                    if name1 in calc.matrices and name2 in calc.matrices:
                        result = calc.subtract_matrices(calc.matrices[name1], calc.matrices[name2])
                        calc.display_matrix(result, f"{name1} - {name2}")
                    else:
                        print("One or both matrices not found!")
                else:
                    print("Need at least 2 stored matrices. Create matrices first.")
                    
            elif choice == '4':
                print("\nMatrix Multiplication: A × B")
                if len(calc.matrices) >= 2:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name1 = input("Enter first matrix name: ").strip()
                    name2 = input("Enter second matrix name: ").strip()
                    
                    if name1 in calc.matrices and name2 in calc.matrices:
                        result = calc.multiply_matrices(calc.matrices[name1], calc.matrices[name2])
                        calc.display_matrix(result, f"{name1} × {name2}")
                    else:
                        print("One or both matrices not found!")
                else:
                    print("Need at least 2 stored matrices. Create matrices first.")
                    
            elif choice == '5':
                print("\nScalar Multiplication")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    scalar = float(input("Enter scalar value: "))
                    
                    if name in calc.matrices:
                        result = calc.scalar_multiply(calc.matrices[name], scalar)
                        calc.display_matrix(result, f"{scalar} × {name}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '6':
                print("\nMatrix Transpose")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        result = calc.transpose_matrix(calc.matrices[name])
                        calc.display_matrix(result, f"Transpose of {name}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '7':
                print("\nDeterminant Calculation")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        det = calc.determinant(calc.matrices[name])
                        print(f"\nDeterminant of {name}: {det:.6f}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '8':
                print("\nMatrix Inverse")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        inverse = calc.inverse_matrix(calc.matrices[name])
                        calc.display_matrix(inverse, f"Inverse of {name}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '9':
                print("\nEigenvalues & Eigenvectors")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        eigenvals, eigenvecs = calc.eigenvalues_eigenvectors(calc.matrices[name])
                        print(f"\nEigenvalues of {name}:")
                        for i, val in enumerate(eigenvals):
                            print(f"  λ{i+1} = {val:.6f}")
                        calc.display_matrix(eigenvecs, f"Eigenvectors of {name}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '10':
                print("\nMatrix Rank")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        rank = calc.rank_matrix(calc.matrices[name])
                        print(f"\nRank of {name}: {rank}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '11':
                print("\nMatrix Trace")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        trace = calc.trace_matrix(calc.matrices[name])
                        print(f"\nTrace of {name}: {trace:.6f}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '12':
                print("\nMatrix Power")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    power = int(input("Enter power: "))
                    
                    if name in calc.matrices:
                        result = calc.power_matrix(calc.matrices[name], power)
                        calc.display_matrix(result, f"{name}^{power}")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '13':
                print("\nSolve Linear System Ax = b")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    a_name = input("Enter coefficient matrix name (A): ").strip()
                    
                    if a_name in calc.matrices:
                        print("Enter constants vector (b):")
                        n = calc.matrices[a_name].shape[0]
                        b_values = input(f"Enter {n} values (space-separated): ").split()
                        b = np.array([float(x) for x in b_values])
                        
                        solution = calc.solve_linear_system(calc.matrices[a_name], b)
                        calc.display_matrix(solution.reshape(-1, 1), "Solution x")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '14':
                print("\nLU Decomposition")
                if calc.matrices:
                    names = list(calc.matrices.keys())
                    print(f"Available matrices: {names}")
                    name = input("Enter matrix name: ").strip()
                    
                    if name in calc.matrices:
                        L, U = calc.lu_decomposition(calc.matrices[name])
                        calc.display_matrix(L, "Lower Triangular (L)")
                        calc.display_matrix(U, "Upper Triangular (U)")
                    else:
                        print("Matrix not found!")
                else:
                    print("No stored matrices. Create a matrix first.")
                    
            elif choice == '15':
                calc.show_history()
                
            elif choice == '16':
                calc.show_stored_matrices()
                
            elif choice == '17':
                calc.clear_history()
                
            elif choice == '18':
                calc.clear_matrices()
                
            elif choice == '19':
                print("\nThank you for using the Matrix Operations Tool!")
                break
                
            else:
                print("Invalid choice! Please enter a number between 1-19.")
                
        except ValueError as e:
            print(f"Error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        input("\nPress Enter to continue...")

def demo_mode():
    """Demonstration mode with predefined examples"""
    print("🔢 MATRIX OPERATIONS TOOL - DEMO MODE")
    print("="*60)
    
    calc = MatrixCalculator()
    
    # Create sample matrices
    print("Creating sample matrices...")
    
    A = np.array([[2, 3], [1, 4]], dtype=float)
    B = np.array([[5, 1], [2, 3]], dtype=float)
    C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    
    calc.matrices['A'] = A
    calc.matrices['B'] = B
    calc.matrices['C'] = C
    
    calc.display_matrix(A, "Matrix A")
    calc.display_matrix(B, "Matrix B")
    calc.display_matrix(C, "Matrix C")
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION OF OPERATIONS")
    print("="*60)
    
    # Addition
    print("\n1. Matrix Addition (A + B):")
    result = calc.add_matrices(A, B)
    calc.display_matrix(result, "A + B")
    
    # Multiplication
    print("\n2. Matrix Multiplication (A × B):")
    result = calc.multiply_matrices(A, B)
    calc.display_matrix(result, "A × B")
    
    # Transpose
    print("\n3. Transpose of A:")
    result = calc.transpose_matrix(A)
    calc.display_matrix(result, "A^T")
    
    # Determinant
    print("\n4. Determinant of A:")
    det = calc.determinant(A)
    print(f"det(A) = {det:.6f}")
    
    # Inverse
    print("\n5. Inverse of A:")
    try:
        inv_A = calc.inverse_matrix(A)
        calc.display_matrix(inv_A, "A^(-1)")
        
        # Verify A × A^(-1) = I
        print("\nVerification (A × A^(-1)):")
        identity = calc.multiply_matrices(A, inv_A)
        calc.display_matrix(identity, "A × A^(-1)")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Eigenvalues and eigenvectors
    print("\n6. Eigenvalues and Eigenvectors of A:")
    try:
        eigenvals, eigenvecs = calc.eigenvalues_eigenvectors(A)
        print("Eigenvalues:")
        for i, val in enumerate(eigenvals):
            print(f"  λ{i+1} = {val:.6f}")
        calc.display_matrix(eigenvecs, "Eigenvectors")
    except Exception as e:
        print(f"Error: {e}")
    
    # Show operation history
    calc.show_history()
    
    print("\n✓ Demo completed!")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Interactive Mode")
    print("2. Demo Mode")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '2':
        demo_mode()
    else:
        interactive_menu()