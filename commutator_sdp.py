import ncpol2sdpa as ncp
from sympy.physics.quantum.dagger import Dagger
import getopt, sys
from math import sqrt

# PATH TO A SOLVER COMPATIBLE WITH THE NCPOL2SDPA PACKAGE
SOLVER_NAME = 'sdpa'
SOLVER_EXE = {'executable' : '/usr/bin/sdpa',
			  'paramsfile' : '/usr/share/sdpa/param.sdpa'}


# Helper function
def commutator(a,b):
	return a*b - b*a

"""
Function that calculates whether there exists a solution to the SDP-relaxed
NC polynomial constraint problem:
	-	||X|| <= c1
	-	||D|| <= c2
	-	||[D,X] - I|| <= eps

Inputs - c1, c2, eps, (sdp)
Returns - true/false, sdp

NOTE: To be computationally economical we shouldn't create new SDPs everytime we
call this function. So the output returns the sdp object which can be passed back
to the function to make evaluation quicker on the second run.
"""
def isFeasible(c1, c2, eps, sdp = None, relaxation_level = 2):
	# No sdp passed, create one
	if sdp is None:
		D = ncp.generate_operators('D', 1)[0]
		X = ncp.generate_operators('X', 1)[0]

		obj = 1.0

		inequality_cons = [c1**2 - Dagger(X)*X >= 0,
						   c2**2 - Dagger(D)*D >= 0,
						   eps**2 - Dagger(commutator(D,X))*commutator(D,X) +
									Dagger(commutator(D,X)) + commutator(D,X) - 1 >= 0]
		sdp = ncp.SdpRelaxation([D,X], normalized = True)
		sdp.get_relaxation(level = relaxation_level,
						   objective = obj,
						   inequalities = inequality_cons)
	else:
		# sdp object passed. Use process_constraints instead
		D = sdp.monomial_sets[0][1]
		X = sdp.monomial_sets[0][2]
		inequality_cons = [c1**2 - Dagger(X)*X >= 0,
						   c2**2 - Dagger(D)*D >= 0,
						   eps**2 - Dagger(commutator(D,X))*commutator(D,X) +
									Dagger(commutator(D,X)) + commutator(D,X) - 1 >= 0]
		sdp.process_constraints(inequalities = inequality_cons)

	# Now solve
	sdp.solve(solver = SOLVER_NAME, solverparameters = SOLVER_EXE)
	if sdp.status == 'optimal':
		return True, sdp
	else:
		return False, sdp

# Function to evaluate whether Popa's theorem considers the tuple of (c1,c2,eps) to be valid
from math import log, exp
def popaCheck(c1,c2,eps):
	if c1*c2 >= -0.5*log(eps):
		return True
	else:
		return False

# Returns the bound on minimum epsilon as asserted by Popa's theorem
def popaMinimum(c1,c2):
	return exp(-2*c1*c2)

# Given c1 and c2 attempts to minimise eps.
def minimalCommutatorDistance(c1, c2, sdp=None, relaxation_level = 2):
	# No sdp passed, create one
	if sdp is None:
		D = ncp.generate_operators('D', 1)[0]
		X = ncp.generate_operators('X', 1)[0]

		# The sdp will ignore the +1 term, we have to add it on at the end
		obj = Dagger(commutator(D,X) - 1)*(commutator(D,X) - 1)

		inequality_cons = [c1**2 - Dagger(X)*X >= 0,
						   c2**2 - Dagger(D)*D >= 0]
		sdp = ncp.SdpRelaxation([D,X], normalized = True)
		sdp.get_relaxation(level = relaxation_level,
						   objective = obj,
						   inequalities = inequality_cons)
	else:
		# sdp object passed. Use process_constraints instead
		D = sdp.monomial_sets[0][1]
		X = sdp.monomial_sets[0][2]
		inequality_cons = [c1**2 - Dagger(X)*X >= 0,
						   c2**2 - Dagger(D)*D >= 0]
		sdp.process_constraints(inequalities = inequality_cons)

	# Now solve
	sdp.solve(solver = SOLVER_NAME, solverparameters = SOLVER_EXE)
	if sdp.status == 'optimal':
		return sqrt(sdp.primal + 1), sdp
	else:
		return None, sdp
"""
Short example of the functionality
"""
if __name__=='__main__':
	import numpy as np

	"""
	Example 1:
		Picks random (c1,c2) and attempts to minimise epsilon
	"""
	# # Initialise sdp
	# _, sdp = minimalCommutatorDistance(1,1)
	# for k in range(10):
	# 	c1, c2 = np.random.random(2)
	# 	eps, _ = minimalCommutatorDistance(c1,c2,sdp)
	# 	print(c1,c2,eps,popaMinimum(c1,c2))

	"""
	Example 2:
		Picks random c1, c2, eps and checks whether a feasible solution exists.
	"""
	# Initialise the sdp by calling isFeasible once
	_, sdp = isFeasible(1, 1, 1, relaxation_level = 2)

	# Let's check 10 random points
	for k in range(10):
		c1, c2, eps = np.random.random(3)
		feasible, _ = isFeasible(c1, c2, eps, sdp)
		popa_feasible = popaCheck(c1,c2,eps)
		if feasible:
			print('(c1,c2,eps) = ({:.2f}, {:.2f}, {:.2f}) is feasible.'.format(c1,c2,eps))
			if popa_feasible:
				print('Popa agrees.')
			else:
				print('Popa disagrees.')
		else:
			print('(c1,c2,eps) = ({:.2f}, {:.2f}, {:.2f}) is not feasible.'.format(c1,c2,eps))
			if not popa_feasible:
				print('Popa agrees.')
			else:
				print('Popa disagrees.')
