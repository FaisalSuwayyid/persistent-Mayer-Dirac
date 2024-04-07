import gudhi
from sympy import Matrix
from sympy import re, im, I, E, pi, symbols
from sympy import Symbol, polar_lift, I
import sympy
import numpy as np
import scipy
import cmath
from scipy.spatial import distance

from biopandas.mol2 import PandasMol2
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import operator
import time




def np_rip_simplicials(point_space, threshold, max_dim):
    # import gudhi
    rips_complex = gudhi.RipsComplex(points = point_space, max_edge_length = threshold)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = max_dim)
    basis = []
    actual_max_dim = 0
    for filtered_value in simplex_tree.get_filtration():
        basis.append(tuple(filtered_value)[0])
        if len(tuple(filtered_value)[0])>actual_max_dim:
            actual_max_dim = len(tuple(filtered_value)[0])

    basis = [sorted([spx for spx in basis if len(spx)==i]) for i in range(1,actual_max_dim+1)]
    
    return basis, actual_max_dim
    
def np_simplex_faces(simplex):
    faces = []
    for i in range(len(simplex)):
        faces.append(simplex[:i]+simplex[i+1:])
    return faces

def np_N_chain_boundary_map(source_basis, target_basis, N = 2):
    # we generally assume if target is zero then source is also zero
    # therefore, we check target first
    if len(target_basis) == 0:
        # B = sympy.zeros(1,1) # please, do a check of a true zero spaces as this is strange in matrices consideration.
        B = np.zeros((1,1), dtype=complex)
    elif len(source_basis) == 0:
        # B = sympy.zeros(len(target_basis),1)
        B = np.zeros((len(target_basis),1), dtype=complex)
    else:
        # B = sympy.zeros(len(target_basis),len(source_basis))
        B = np.zeros((len(target_basis),len(source_basis)), dtype=complex)
        source_dim = len(source_basis[0])
        for i in range(len(source_basis)):
            faces = np_simplex_faces(source_basis[i])
            for k in range(source_dim):
                # B[target_basis.index(faces[k]), i] = E**(k*2*pi*I/N)
                B[target_basis.index(faces[k]), i] = np.exp((2 * cmath.pi * 1j * k) / N)
    return B


    
def np_N_chain_complex(simplices_basis, max_dimension = 0, N = 2):
    l = len(simplices_basis)
    n = min(l - 1, max_dimension)
    N_chain = []
    # N_chain.append(sympy.zeros(1,len(simplices_basis[0]))) # assuming max_dimension>=0
    N_chain.append(np.zeros((1,len(simplices_basis[0])), dtype=complex))
    if n > 0:
        for i in range(n):
            N_chain.append(np_N_chain_boundary_map(simplices_basis[i+1], simplices_basis[i], N))
    if max_dimension - n > 0:
        # N_chain.append(sympy.zeros(len(simplices_basis[n]), 1))
        N_chain.append(np.zeros((len(simplices_basis[n]), 1), dtype=complex))
        for i in range(n + 2, max_dimension + 1):
            # N_chain.append(sympy.zeros(1, 1))
            N_chain.append(np.zeros((1, 1), dtype=complex))
    return N_chain
        

def np_Embedding_subbasis(source_basis, target_basis):
    # we generally assume if target is zero then source is also zero
    # therefore, we check target first
    if len(target_basis) == 0:
        # Embedding_map = sympy.zeros(1,1) # please, do a check of a true zero spaces as this is strange in matrices consideration.
        Embedding_map = np.zeros((1,1), dtype=complex)
    elif len(source_basis) == 0:
        # Embedding_map = sympy.zeros(len(target_basis),1)
        Embedding_map = np.zeros((len(target_basis),1), dtype=complex)
    else:
        # Embedding_map = sympy.zeros(len(target_basis),len(source_basis))
        Embedding_map = np.zeros((len(target_basis),len(source_basis)), dtype=complex)
        for i in range(len(source_basis)):
            Embedding_map[target_basis.index(source_basis[i]), i] = 1
    return Embedding_map




def np_N_chain_embeddings(simplices_basis1, simplices_basis2, max_dimension = 0):
    # we assume simplices_basis1 is subset of simplices_basis2 in general
    # therefore, we do the embeddings up to min(l1, max_dim), and then fill rest with zeros
    # we assume in general both have nonempty vertices
    l1 = len(simplices_basis1)
    l2 = len(simplices_basis2)
    n = min(l1 - 1, max_dimension) # to do the source non-zero spaces first
    m = min(l2 - 1, max_dimension) # to do the target non-zero spaces first
    embeddings = []
    for i in range(n + 1):
        embeddings.append(np_Embedding_subbasis(simplices_basis1[i], simplices_basis2[i]))
    for i in range(n + 1, m + 1):
        embeddings.append(sympy.zeros(len(simplices_basis2[i]), 1))
    for i in range(m + 1, max_dimension + 1):
        embeddings.append(sympy.zeros(1, 1))
    return embeddings



def np_N_chain_embeddings(simplices_basis1, simplices_basis2, max_dimension = 0):
    # we assume simplices_basis1 is subset of simplices_basis2 in general
    # therefore, we do the embeddings up to min(l1, max_dim), and then fill rest with zeros
    # we assume in general both have nonempty vertices
    l1 = len(simplices_basis1)
    l2 = len(simplices_basis2)
    n = min(l1 - 1, max_dimension) # to do the source non-zero spaces first
    m = min(l2 - 1, max_dimension) # to do the target non-zero spaces first
    embeddings = []
    for i in range(n + 1):
        embeddings.append(np_Embedding_subbasis(simplices_basis1[i], simplices_basis2[i]))
    for i in range(n + 1, m + 1):
        # embeddings.append(sympy.zeros(len(simplices_basis2[i]), 1))
        embeddings.append(np.zeros((len(simplices_basis2[i]), 1), dtype=complex))
    for i in range(m + 1, max_dimension + 1):
        # embeddings.append(sympy.zeros(1, 1))
        embeddings.append(np.zeros((1, 1), dtype=complex))
    return embeddings
    

def np_check_commutativity(chain1, chain2, embedding):
    for i in range(1, len(embedding)):
        print(embedding[i-1]@chain1[i] == chain2[i]@embedding[i])
    return None



########################################################################
########################################################################
########################################################################
########################################################################


###### numerical considerations are being taken in this part

def np_Intersection_Of_Spaces(ASpan, BSpan): 
    # both spaces are assumed to be not zero spaces.
    # please handle these cases separately.
    """
      Compute the basis of the intersection of two vector spaces given their generators.

      Args:
          ASpan (matrix of generators, column form): Basis of the first vector space.
          BSpan (same): Basis of the second vector space.

      Returns:
          matrix: Basis of the intersection of the two vector spaces in the first space.
      """
    # return basis of intersection of two spaces ASpan and BSpan, where these two are generating vectors
    # The return basis resides in ASpan space
    # A_matrix = sympy.Matrix(ASpan)
    A_matrix = np.array(ASpan)
    # Acols = A_matrix.cols
    Acols = A_matrix.shape[1]
    # B_matrix = sympy.Matrix(BSpan)
    B_matrix = np.array(BSpan)
    # Intersection_Matrix = A_matrix.row_join(-B_matrix)
    Intersection_Matrix = np.concatenate((A_matrix, -B_matrix), axis = 1)
    # Intersection_Matrix = sympy.N(A_matrix.row_join(-B_matrix)) # sympy.N for numerical calcluations since expressions become too big for symbolic calculations
    # NullAuxiliary = Intersection_Matrix.nullspace()
    NullAuxiliary = scipy.linalg.null_space(Intersection_Matrix)
    if NullAuxiliary.size > 0: # check if null_space is not zero space
        # ColumnsOutOfNullBasis = sympy.Matrix([[NullAuxiliary[i][j] for j in range(Acols)] for i in range(len(NullAuxiliary))]) # span of nullspace of A
        ColumnsOutOfNullBasis = NullAuxiliary[:Acols, :]
        # BasisAuxiliary = ColumnsOutOfNullBasis.rref()[0]
        # BasisAuxiliary = ColumnsOutOfNullBasis.LUdecomposition[1].rref()[0] # way faster than just rref, like way way faster
        BasisAuxiliary = np.linalg.qr(ColumnsOutOfNullBasis, mode='complete')[0]
        return BasisAuxiliary
    # return sympy.zeros(Acols, 1)
    return np.zeros((Acols, 1), dtype=complex)


def np_middle_complex_embeddings(simplices_basis2, chain2, embeddings):
    middle_embeddings = []
    # middle_embeddings.append(sympy.eye(chain2[0].cols))
    middle_embeddings.append(np.eye(chain2[0].shape[1]))
    for i in range(1, len(simplices_basis2)): # non-zero spaces Bn, of An -> Bn
        middle_embeddings.append(np_Intersection_Of_Spaces(chain2[i], embeddings[i-1]))
    for i in range(len(simplices_basis2), len(chain2)): # zero spaces
        # middle_embeddings.append(sympy.zeros(1))
        middle_embeddings.append(np.zeros((1, 1)))
    return middle_embeddings



########################################################################
########################################################################
########################################################################
########################################################################



def np_embeddings_orthonormalization(embeddings):
    for i in range(len(embeddings)):
        # if sympy.det(embeddings[i].T*embeddings[i]) != 0:
        if np.abs(np.linalg.det(embeddings[i].T@embeddings[i])) > 0.0001:
            # embeddings[i] = embeddings[i].QRdecomposition()[0]
            embeddings[i] = np.linalg.qr(embeddings[i], mode='complete')[0]
    return embeddings

def np_middle_comlex_boundary_maps(basis2, chain2, middle_embeddings):
    boundary_maps = []
    # boundary_maps.append(sympy.zeros(1, middle_embeddings[0].rows))
    boundary_maps.append(np.zeros((1, middle_embeddings[0].shape[0]), dtype=complex))
    for i in range(1, len(basis2)): # non-zero spaces first
        boundary_maps.append(np.linalg.inv(middle_embeddings[i-1].conjugate().T@middle_embeddings[i-1])@middle_embeddings[i-1].conjugate().T@chain2[i]@middle_embeddings[i])
    for i in range(len(basis2), len(chain2)):
        # boundary_maps.append(sympy.zeros(middle_embeddings[i-1].rows, middle_embeddings[i].rows)) # row = target, column = source
        boundary_maps.append(np.zeros((middle_embeddings[i-1].shape[0], middle_embeddings[i].shape[0]), dtype=complex))
    return boundary_maps
    
def np_q_th_sequence(N, q, length, sequence = [], seq = 0):
    seq += q
    if seq < length:
        sequence.append(seq)
        np_q_th_sequence(N, N - q, length, sequence, seq)
    return sequence

def np_sublist_of_list_given_index(l, index):
    return list(operator.itemgetter(*index)(l))

def np_sequential_composition(chain, n, m): # n <= m
    A = chain[n + 1]
    for i in range(n + 2, m + 1):
        # A *= chain[i]
        A = A@chain[i]
    return A

def np_q_th_subchain(chain, N, t, q):
    index = np_q_th_sequence(N, q, length = len(chain), sequence = [t], seq = t)
    subchain = []
    subchain.append(np.zeros((1, chain[t].shape[1])))
    for i in range(len(index) - 1):
        subchain.append(np_sequential_composition(chain, index[i], index[i+1]))
    return subchain

def np_chain_laplacians(chain): # assuming orthonormal basis
    laplacians = []
    for i in range(len(chain) - 1):
        # laplacians.append(chain[i].H*chain[i] + chain[i+1]*chain[i+1].H) # down + up
        laplacians.append(chain[i].conjugate().T@chain[i] + chain[i+1]@chain[i+1].conjugate().T)
    # laplacians.append(chain[len(chain) - 1].H*chain[len(chain) - 1]) # last is just down
    laplacians.append(chain[len(chain) - 1].conjugate().T@chain[len(chain) - 1]) # last is just down
    return laplacians

def np_Initial_Dirac(boundary1):
    # A = sympy.zeros(boundary1.rows, boundary1.rows).row_join(boundary1) # [0 B]
    A = np.concatenate((np.zeros((boundary1.shape[0], boundary1.shape[0]), dtype=complex),boundary1), axis = 1) # [0 B]
    # C = boundary1.H.row_join(sympy.zeros(boundary1.cols, boundary1.cols)) # [B* 0]
    C = np.concatenate((boundary1.conjugate().T, np.zeros((boundary1.shape[1], boundary1.shape[1]), dtype=complex)), axis = 1) # [B* 0]
    # D = A.col_join(C) # Dirac0
    D = np.concatenate((A, C), axis = 0)
    return D


def np_Next_Dirac(Previous_Dirac, Next_Boundary):
    # 1
    # C = sympy.zeros(Previous_Dirac.rows - Next_Boundary.rows, Next_Boundary.cols).col_join(Next_Boundary) # [[0], [B]]
    C = np.concatenate((np.zeros((Previous_Dirac.shape[0] - Next_Boundary.shape[0], Next_Boundary.shape[1]), dtype=complex), Next_Boundary), axis = 0) # [[0], [B]]
    # 2
    # D = Previous_Dirac.row_join(C) # [Previous_Dirac C]
    D = np.concatenate((Previous_Dirac, C), axis = 1) # [Previous_Dirac C]
    # 3
    # F = sympy.zeros(Next_Boundary.H.rows, Previous_Dirac.cols-Next_Boundary.H.cols).row_join(Next_Boundary.H) # [0 E]
    F = np.concatenate((np.zeros((Next_Boundary.conjugate().T.shape[0], Previous_Dirac.shape[1]-Next_Boundary.conjugate().T.shape[1]), dtype=complex), Next_Boundary.conjugate().T), axis = 1) # [0 E]
    # 4
    # G = F.row_join(sympy.zeros(Next_Boundary.cols, Next_Boundary.cols)) # [F 0] = [0 E 0]
    G = np.concatenate((F, np.zeros((Next_Boundary.shape[1], Next_Boundary.shape[1]), dtype=complex)), axis = 1) # [F 0] = [0 E 0]
    # 5
    # H = D.col_join(G) # [[D], [G]]
    H = np.concatenate((D, G), axis = 0) # [[D], [G]]
    return H


def np_chain_diracs(chain): # include the first zero map
    Length = len(chain)
    if Length == 1:
        B = np.zeros((chain[0].shape[1], 1))
        D = [[]]
        D[0] = np_Initial_Dirac(B)
        return D
    D = [[] for i in range(Length - 1)]
    D[0] = np_Initial_Dirac(chain[1])
    for i in range(1, Length - 1):
        D[i] = np_Next_Dirac(D[i-1], chain[i+1])
    return D


def np_chain_laplacians_diracs(chain):
    laplacians = [[] for i in range(len(chain)-1)]
    for i in range(len(chain)-1):
        laplacians[i] = np_chain_laplacians(chain[:i+2])
    return laplacians

def np_sympy_to_numpy_matrix(A):
    # return np.array([[complex(A[i,j]) for j in range(A.cols)] for i in range(A.rows)])
    return np.array([[complex(A[i,j]) for j in range(A.shape[1])] for i in range(A.shape[0])])



def np_sequence_eigenvalues(sequence): # we also take the real part only since we are assuming dealing with hermitian, thus cutting of the imaginary
    buffer = []
    for i in range(len(sequence)):
        buffer.append(np_sympy_to_numpy_matrix(sequence[i]))
    seq_eigvals = []
    for i in range(len(sequence)):
        # seq_eigvals.append(np.linalg.eigvalsh(sequence[i])) # symmetric real or hermitian
        seq_eigvals.append(np.round(np.real(np.linalg.eigvals(buffer[i])), decimals = 3)) # general
    return seq_eigvals


def np_sequence_betti(basis, sequence): # usually we pass laplacians and diracs which are hermitian, thus faster algorithm
    buffer = []
    for i in range(len(sequence)):
        buffer.append(np_sympy_to_numpy_matrix(sequence[i]))
    seq_betti = []
    for i in range(len(sequence)):
        if basis[i] == 0:
            seq_betti.append(0)
        else:
            seq_betti.append(buffer[i].shape[1] - np.linalg.matrix_rank(buffer[i], hermitian=True))
    return seq_betti
    
    
def np_rounding_off_sympy(mat, tol):
    buffer = mat
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            # if sympy.functions.elementary.complexes.Abs(mat[i,j]) < tol:
            # if np.abs(mat[i,j]) < tol:
                # print(np.abs(mat[i,j]))
                # mat[i,j] = 0
            buffer[i,j] = np.round(buffer[i,j], decimals = 3)
    return buffer


def np_eigen_means(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_mean = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            eg_mean.append(np.mean(a))
        else:
            eg_mean.append(0)
    return eg_mean

def np_eigen_sum(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_sum = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            eg_sum.append(np.sum(a))
        else:
            eg_sum.append(0)
    return eg_sum


def np_eigen_min(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_min = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            eg_min.append(np.min(a))
        else:
            eg_min.append(0)
    return eg_min

def np_eigen_max(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_max = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            eg_max.append(np.max(a))
        else:
            eg_max.append(0)
    return eg_max

def np_eigen_std(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_std = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            eg_std.append(np.std(a))
        else:
            eg_std.append(0)
    return eg_std

def np_dirac_signless_Euler_Poincare(bases_dims, bettis):
    x = np.array(bases_dims) - np.array(bettis)
    EP = []
    for i in range(0, len(bases_dims)):
        EP.append(0.5*np.sum(x[:i+1]))
    return EP[:len(EP)-1]

def dirac_zeta(s, dirac_eigenvalues):
    zeta = []
    for x in dirac_eigenvalues:
        a = x[x>0.0001]
        if a.size > 0:
            zeta.append(np.sum(a**(-s)))
        else:
            zeta.append(0)
    return zeta

def s_dirac_complexity(s, dirac_eigenvalues):
    # product becomes too big, take log already
    c = []
    for x in dirac_eigenvalues:
        a = x[x>0.0001]
        if a.size > 0:
            c.append(np.sum(np.log(a**(2*s))))
        else:
            c.append(0)
    return c

def spanning_tree_number(first_dirac_complexity, dirac_euler_poincare):
    t = []
    for x, y in zip(first_dirac_complexity, dirac_euler_poincare):
        t.append(0.5*x - np.log(1+y))
    return t

def Quasi_Wiener(eigen_sequence):
    QW = []
    buffer = s_dirac_complexity(1, eigen_sequence)
    for x, y in zip(eigen_sequence, buffer):
        a = x[x>0.0001]
        if a.size > 0:
            QW.append(y*(a.size + 1))
        else:
            QW.append(0)
    return QW
    
    

def np_eigen_positive_generalized_mean(eigen_sequence):
    # python list, each row is a numpy array of eigen values of an operator
    # we compute the mean of the positive eigen values.
    eg_pos_gen_mean = []
    for x in eigen_sequence:
        a = x[x>0.0001]
        if a.size > 0:
            mu = np.mean(a)
            eg_pos_gen_mean.append(np.mean(np.abs(a - mu)))
        else:
            eg_pos_gen_mean.append(0)
    return eg_pos_gen_mean

def np_N_chain_complex_dims(simplices_basis, max_dimension = 0, N = 2):
    l = len(simplices_basis)
    n = min(l - 1, max_dimension)
    N_chain_dims = []
    N_chain_dims.append(len(simplices_basis[0]))
    if n > 0:
        for i in range(n):
            N_chain_dims.append(len(simplices_basis[i+1]))
    if max_dimension - n > 0:
        # N_chain.append(sympy.zeros(len(simplices_basis[n]), 1))
        N_chain_dims.append(0)
        for i in range(n + 2, max_dimension + 1):
            # N_chain.append(sympy.zeros(1, 1))
            N_chain_dims.append(0)
    return N_chain_dims


def np_filtration(points, num_filtrations = 1, max_radius = 10, max_dimension = 1, N = 2):
    filts = []
    if num_filtrations > 1:
        radii = np.linspace(0, max_radius, num_filtrations, endpoint=True)
    else:
        radii = np.array([max_radius])
    for rad in radii:
        basis1, _ = np_rip_simplicials(points, rad, max_dimension)
        chain1 = np_N_chain_complex(basis1, max_dimension = max_dimension, N = N)
        filts.append(chain1)
    return filts, radii, basis1

def np_filtration_features(filts, N = 2, q = 1, trim = 0):
    laps_features = {}
    diracs_features = {}
    for chain in filts:
        chain = chain[trim:]
        chain1 = np_q_th_subchain(chain, N, q)
        
        laps = np_chain_laplacians(chain1)
        laps_features['betti'] = np_sequence_betti(laps)
        laps_features['eigenvalues'] = np_sequence_eigenvalues(laps)
        
        laps_features['mean'] = np_eigen_means(laps_features['eigenvalues'])
        laps_features['max'] = np_eigen_max(laps_features['eigenvalues'])
        laps_features['min'] = np_eigen_min(laps_features['eigenvalues'])
        laps_features['generalized_mean'] = np_eigen_positive_generalized_mean(laps_features['eigenvalues'])
        
        diracs = np_chain_diracs(chain1)
        diracs_features['betti'] = np_sequence_betti(diracs)
        diracs_features['eigenvalues'] = np_sequence_eigenvalues(diracs)
        
        diracs_features['mean'] = np_eigen_means(diracs_features['eigenvalues'])
        diracs_features['max'] = np_eigen_max(diracs_features['eigenvalues'])
        diracs_features['min'] = np_eigen_min(diracs_features['eigenvalues'])
        diracs_features['generalized_mean'] = np_eigen_positive_generalized_mean(diracs_features['eigenvalues'])
    return laps_features, diracs_features


def np_filtration(points, num_filtrations = 1, max_radius = 10, max_dimension = 1, N = 2):
    filts = []
    
    filts_bettis = []
    filts_dirac_bettis = []
    filts_means = []
    filts_sums = []
    filts_mins = []
    filts_maxs = []
    filts_stds = []
    filts_gen_means = []
    filts_sec_moms = []
    filts_zets2 = []
    filts_zets1 = []
    filts_c1 = []
    filts_EPs = []
    filts_stns = []
    filts_QWs = []
    
    d_filts_bettis = []
    d_filts_dirac_bettis = []
    d_filts_means = []
    d_filts_sums = []
    d_filts_mins = []
    d_filts_maxs = []
    d_filts_stds = []
    d_filts_gen_means = []
    d_filts_sec_moms = []
    d_filts_zets2 = []
    d_filts_zets1 = []
    # d_filts_c1 = []
    # d_filts_EPs = []
    # d_filts_stns = []
    d_filts_QWs = []
    
    
    
    if num_filtrations > 1:
        radii = np.linspace(0, max_radius, num_filtrations, endpoint=True)
    else:
        radii = np.array([max_radius])
    for rad in radii:
        # print("calculate chains")
        np_basis1, actual = np_rip_simplicials(points, rad, max_dimension)
        # print(np_basis1)
        np_chain1 = np_N_chain_complex(np_basis1, max_dimension = max_dimension, N = N)
        # print(np_chain1)
        
        feature_bettis = []
        feature_means = []
        feature_sums = []
        feature_mins = []
        feature_maxs = []
        feature_stds = []
        feature_gen_means = []
        feature_sec_moms = []
        feature_zets2 = []
        feature_zets1 = []
        feature_c1 = []
        feature_EPs = []
        feature_stns = []
        feature_QWs = []
        
        
        d_feature_bettis = []
        d_feature_means = []
        d_feature_sums = []
        d_feature_mins = []
        d_feature_maxs = []
        d_feature_stds = []
        d_feature_gen_means = []
        d_feature_sec_moms = []
        d_feature_zets2 = []
        d_feature_zets1 = []
        # d_feature_c1 = []
        # d_feature_EPs = []
        # d_feature_stns = []
        d_feature_QWs = []
        
    
        for i in range(N):
            for q in range(1, N):
                # print("calculate subchains")
                dims_bases = np_N_chain_complex_dims(np_basis1, max_dimension = max_dimension, N = N)
                # print(np_q_th_sequence(N, q, length = len(np_chain1), sequence = [i], seq = i))
                subchain_bases_dims = [dims_bases[j] for j in np_q_th_sequence(N, q, length = len(np_chain1), sequence = [i], seq = i)]
                subchain = np_q_th_subchain(np_chain1, N, t = i, q = q)
                # print("calculate laps")
                laps = np_chain_laplacians(subchain)
                # print(laps)

                # print("calculate bettis")
                bettis = np_sequence_betti(subchain_bases_dims, laps)
                # features = features + [bettis]
                # features = features + bettis
                feature_bettis = feature_bettis + [bettis]
                d_feature_bettis = d_feature_bettis + [[np.sum(np.array(bettis))]]
                # print(bettis)

                eigens = np_sequence_eigenvalues(laps)
                # print(eigens)
                dirac_eigens = []
                dirac_eigens.append(np.array([np.sqrt(eg) for s in np_sequence_eigenvalues(laps) for eg in s]))

                means = np_eigen_means(eigens)
                # features = features + [means]
                # features = features + means
                feature_means = feature_means + [means]

                sums = np_eigen_sum(eigens)
                # features = features + [sums]
                # features = features + sums
                feature_sums = feature_sums + [sums]

                mins = np_eigen_min(eigens)
                # features = features + [mins]
                # features = features + mins
                feature_mins = feature_mins + [mins]

                maxes = np_eigen_max(eigens)
                # features = features + [maxes]
                # features = features + maxes
                feature_maxs = feature_maxs + [maxes]

                stds = np_eigen_std(eigens)
                # features = features + [stds]
                # features = features + stds
                feature_stds = feature_stds + [stds]

                gen_means = np_eigen_positive_generalized_mean(eigens)
                # features = features + [gen_means]
                # features = features + gen_means
                feature_gen_means = feature_gen_means + [gen_means]

                sec_moms = dirac_zeta(-2, eigens)
                # features = features + [sec_moms]
                # features = features + sec_moms
                feature_sec_moms = feature_sec_moms + [sec_moms]

                zet2s = dirac_zeta(2, eigens)
                # features = features + [zet2s]
                # features = features + zet2s
                feature_zets2 = feature_zets2 +[zet2s]

                zet1s = dirac_zeta(1, eigens)
                # features = features + [zet1s]
                # features = features + zet1s
                feature_zets1 = feature_zets1 + [zet1s]

                c1 = s_dirac_complexity(1, eigens)
                # features = features + [c1]
                # features = features + c1
                feature_c1 = feature_c1 + [c1]
                
                EPs = np_dirac_signless_Euler_Poincare(subchain_bases_dims, bettis)
                # features = features + [EPs]
                # features = features + EPs
                feature_EPs = feature_EPs + [EPs]

                stns = spanning_tree_number(c1, EPs)
                # features = features + [stns]
                # features = features + stns
                feature_stns = feature_stns + [stns]

                QWs = Quasi_Wiener(eigens)
                # features = features + [QWs]
                # features = features + QWs
                feature_QWs = feature_QWs + [QWs]
                
                
#                 ######## Dirac ones
#                 # print("calculate eigens")
#                 # diracs = np_chain_diracs(subchain)
#                 dirac_laps = np_chain_laplacians_diracs(subchain)
#                 dirac_bettis = []
#                 dirac_eigens = []
#                 for x in dirac_laps:
#                     dirac_bettis.append(np_sequence_betti(subchain_bases_dims[:len(x)], x))
#                     dirac_eigens.append(np.array([np.sqrt(eg) for s in np_sequence_eigenvalues(x) for eg in s]))
#                 # bettis = np_sequence_betti(diracs)
#                 # eigens = np_sequence_eigenvalues(diracs)
                
#                 d_feature_bettis = d_feature_bettis + dirac_bettis
                
               
                
                dmeans = np_eigen_means(dirac_eigens)
                # features = features + [means]
                # features = features + means
                d_feature_means = d_feature_means + [dmeans]

                dsums = np_eigen_sum(dirac_eigens)
                # features = features + [sums]
                # features = features + sums
                d_feature_sums = d_feature_sums + [dsums]

                dmins = np_eigen_min(dirac_eigens)
                # features = features + [mins]
                # features = features + mins
                d_feature_mins = d_feature_mins + [dmins]

                dmaxes = np_eigen_max(dirac_eigens)
                # features = features + [maxes]
                # features = features + maxes
                d_feature_maxs = d_feature_maxs + [dmaxes]

                dstds = np_eigen_std(dirac_eigens)
                # features = features + [stds]
                # features = features + stds
                d_feature_stds = d_feature_stds + [dstds]

                dgen_means = np_eigen_positive_generalized_mean(dirac_eigens)
                # features = features + [gen_means]
                # features = features + gen_means
                d_feature_gen_means = d_feature_gen_means + [dgen_means]

                dsec_moms = dirac_zeta(-2, dirac_eigens)
                # features = features + [sec_moms]
                # features = features + sec_moms
                d_feature_sec_moms = d_feature_sec_moms + [dsec_moms]

                dzet2s = dirac_zeta(2, dirac_eigens)
                # features = features + [zet2s]
                # features = features + zet2s
                d_feature_zets2 = d_feature_zets2 +[dzet2s]

                dzet1s = dirac_zeta(1, dirac_eigens)
                # features = features + [zet1s]
                # features = features + zet1s
                d_feature_zets1 = d_feature_zets1 + [dzet1s]

# #                 c1 = s_dirac_complexity(1, dirac_eigens)
# #                 # features = features + [c1]
# #                 # features = features + c1
# #                 d_feature_c1 = d_feature_c1 + [c1]
                
# #                 EPs = np_dirac_signless_Euler_Poincare(subchain_bases_dims, bettis)
# #                 # features = features + [EPs]
# #                 # features = features + EPs
# #                 d_feature_EPs = d_feature_EPs + [EPs]

# #                 stns = spanning_tree_number(c1, EPs)
# #                 # features = features + [stns]
# #                 # features = features + stns
# #                 d_feature_stns = d_feature_stns + [stns]

                dQWs = Quasi_Wiener(dirac_eigens)
                # features = features + [QWs]
                # features = features + QWs
                d_feature_QWs = d_feature_QWs + [dQWs]
                
        
        filts_bettis = filts_bettis + [feature_bettis]
        filts_means = filts_means + [feature_means]
        filts_sums = filts_sums + [feature_sums]
        filts_mins = filts_mins + [feature_mins]
        filts_maxs = filts_maxs + [feature_maxs]
        filts_stds = filts_stds + [feature_stds]
        filts_gen_means = filts_gen_means + [feature_gen_means]
        filts_sec_moms = filts_sec_moms + [feature_sec_moms]
        filts_zets2 = filts_zets2 + [feature_zets2]
        filts_zets1 = filts_zets1 + [feature_zets1]
        filts_c1 = filts_c1 + [feature_c1]
        filts_EPs = filts_EPs + [feature_EPs]
        filts_stns = filts_stns + [feature_stns]
        filts_QWs = filts_QWs + [feature_QWs]
        
        d_filts_bettis = d_filts_bettis + [d_feature_bettis]
        d_filts_means = d_filts_means + [d_feature_means]
        d_filts_sums = d_filts_sums + [d_feature_sums]
        d_filts_mins = d_filts_mins + [d_feature_mins]
        d_filts_maxs = d_filts_maxs + [d_feature_maxs]
        d_filts_stds = d_filts_stds + [d_feature_stds]
        d_filts_gen_means = d_filts_gen_means + [d_feature_gen_means]
        d_filts_sec_moms = d_filts_sec_moms + [d_feature_sec_moms]
        d_filts_zets2 = d_filts_zets2 + [d_feature_zets2]
        d_filts_zets1 = d_filts_zets1 + [d_feature_zets1]
        # d_filts_c1 = []
        # d_filts_EPs = []
        # d_filts_stns = []
        d_filts_QWs = d_filts_QWs + [d_feature_QWs]
        
        

      
    
    # return None
    # return [radii, filts_bettis,\
    #         filts_means, filts_sums, filts_mins,\
    #         filts_maxs, filts_stds, filts_gen_means,\
    #         filts_sec_moms, filts_zets2, filts_zets1,\
    #         filts_c1, filts_EPs, filts_stns, filts_QWs, d_filts_bettis]
    return [radii, filts_bettis,\
            filts_means, filts_sums, filts_mins,\
            filts_maxs, filts_stds, filts_gen_means,\
            filts_sec_moms, filts_zets2, filts_zets1,\
            filts_c1, filts_EPs, filts_stns, filts_QWs,\
            d_filts_bettis, d_filts_means, d_filts_sums,\
            d_filts_mins, d_filts_maxs, d_filts_stds,\
            d_filts_gen_means, d_filts_sec_moms, d_filts_zets2,\
            d_filts_zets1, d_filts_QWs]



def flattener(fea_seq, fil_length = 10):
    a = np.empty((fil_length,0), dtype=float)
    for fea in fea_seq:
        seqflat = []
        for y in range(len(fea)):
            flat = []
            for x in fea[y]:
                flat = flat + x
            seqflat = seqflat + [flat]
        a = np.concatenate((a, np.array(seqflat)), axis = 1)
    return a

def flattener_indivisual(fea_seq, fil_length):
    a = [[] for i in range(len(fea_seq))]

    for i, fea in zip(range(len(fea_seq)), fea_seq):
        seqflat = []
        for y in range(len(fea)):
            flat = []
            for x in fea[y]:
                flat = flat + x
            seqflat = seqflat + [flat]
        a[i] = np.array(seqflat)
    return a




# betti plots
# betti heat map plots
# eigenvalues plots
# eigenvalues heatplot






def np_filtration_weighted(points, num_filtrations = 1, max_radius = 10, max_dimension = 1, N = 2):
    filts = []
    
    filts_bettis = []
    filts_dirac_bettis = []
    filts_means = []
    filts_sums = []
    filts_mins = []
    filts_maxs = []
    filts_stds = []
    filts_gen_means = []
    filts_sec_moms = []
    filts_zets2 = []
    filts_zets1 = []
    filts_c1 = []
    filts_EPs = []
    filts_stns = []
    filts_QWs = []
    
    d_filts_bettis = []
    d_filts_dirac_bettis = []
    d_filts_means = []
    d_filts_sums = []
    d_filts_mins = []
    d_filts_maxs = []
    d_filts_stds = []
    d_filts_gen_means = []
    d_filts_sec_moms = []
    d_filts_zets2 = []
    d_filts_zets1 = []
    # d_filts_c1 = []
    # d_filts_EPs = []
    # d_filts_stns = []
    d_filts_QWs = []
    
    
    
    if num_filtrations > 1:
        radii = np.linspace(0, max_radius, num_filtrations, endpoint=True)
    else:
        radii = np.array([max_radius])
    for rad in radii:
        # print("calculate chains")
        np_basis1, actual = np_rip_simplicials(points, rad, max_dimension)
        # print(np_basis1)
        np_chain1 = np_N_chain_complex(np_basis1, max_dimension = max_dimension, N = N)
        # print(np_chain1)
        wghts = weights(np_basis1, points)
        np_chain1 = weighted_chain(np_chain1, wghts)
        
        feature_bettis = []
        feature_means = []
        feature_sums = []
        feature_mins = []
        feature_maxs = []
        feature_stds = []
        feature_gen_means = []
        feature_sec_moms = []
        feature_zets2 = []
        feature_zets1 = []
        feature_c1 = []
        feature_EPs = []
        feature_stns = []
        feature_QWs = []
        
        
        d_feature_bettis = []
        d_feature_means = []
        d_feature_sums = []
        d_feature_mins = []
        d_feature_maxs = []
        d_feature_stds = []
        d_feature_gen_means = []
        d_feature_sec_moms = []
        d_feature_zets2 = []
        d_feature_zets1 = []
        # d_feature_c1 = []
        # d_feature_EPs = []
        # d_feature_stns = []
        d_feature_QWs = []
        
    
        for i in range(N):
            for q in range(1, N):
                # print("calculate subchains")
                dims_bases = np_N_chain_complex_dims(np_basis1, max_dimension = max_dimension, N = N)
                # print(np_q_th_sequence(N, q, length = len(np_chain1), sequence = [i], seq = i))
                subchain_bases_dims = [dims_bases[j] for j in np_q_th_sequence(N, q, length = len(np_chain1), sequence = [i], seq = i)]
                subchain = np_q_th_subchain(np_chain1, N, t = i, q = q)
                # print("calculate laps")
                laps = np_chain_laplacians(subchain)
                # print(laps)

                # print("calculate bettis")
                bettis = np_sequence_betti(subchain_bases_dims, laps)
                # features = features + [bettis]
                # features = features + bettis
                feature_bettis = feature_bettis + [bettis]
                d_feature_bettis = d_feature_bettis + [[np.sum(np.array(bettis))]]
                # print(bettis)

                eigens = np_sequence_eigenvalues(laps)
                # print(eigens)
                dirac_eigens = []
                dirac_eigens.append(np.array([np.sqrt(eg) for s in np_sequence_eigenvalues(laps) for eg in s]))

                means = np_eigen_means(eigens)
                # features = features + [means]
                # features = features + means
                feature_means = feature_means + [means]

                sums = np_eigen_sum(eigens)
                # features = features + [sums]
                # features = features + sums
                feature_sums = feature_sums + [sums]

                mins = np_eigen_min(eigens)
                # features = features + [mins]
                # features = features + mins
                feature_mins = feature_mins + [mins]

                maxes = np_eigen_max(eigens)
                # features = features + [maxes]
                # features = features + maxes
                feature_maxs = feature_maxs + [maxes]

                stds = np_eigen_std(eigens)
                # features = features + [stds]
                # features = features + stds
                feature_stds = feature_stds + [stds]

                gen_means = np_eigen_positive_generalized_mean(eigens)
                # features = features + [gen_means]
                # features = features + gen_means
                feature_gen_means = feature_gen_means + [gen_means]

                sec_moms = dirac_zeta(-2, eigens)
                # features = features + [sec_moms]
                # features = features + sec_moms
                feature_sec_moms = feature_sec_moms + [sec_moms]

                zet2s = dirac_zeta(2, eigens)
                # features = features + [zet2s]
                # features = features + zet2s
                feature_zets2 = feature_zets2 +[zet2s]

                zet1s = dirac_zeta(1, eigens)
                # features = features + [zet1s]
                # features = features + zet1s
                feature_zets1 = feature_zets1 + [zet1s]

                c1 = s_dirac_complexity(1, eigens)
                # features = features + [c1]
                # features = features + c1
                feature_c1 = feature_c1 + [c1]
                
                EPs = np_dirac_signless_Euler_Poincare(subchain_bases_dims, bettis)
                # features = features + [EPs]
                # features = features + EPs
                feature_EPs = feature_EPs + [EPs]

                stns = spanning_tree_number(c1, EPs)
                # features = features + [stns]
                # features = features + stns
                feature_stns = feature_stns + [stns]

                QWs = Quasi_Wiener(eigens)
                # features = features + [QWs]
                # features = features + QWs
                feature_QWs = feature_QWs + [QWs]
                
                
#                 ######## Dirac ones
#                 # print("calculate eigens")
#                 # diracs = np_chain_diracs(subchain)
#                 dirac_laps = np_chain_laplacians_diracs(subchain)
#                 dirac_bettis = []
#                 dirac_eigens = []
#                 for x in dirac_laps:
#                     dirac_bettis.append(np_sequence_betti(subchain_bases_dims[:len(x)], x))
#                     dirac_eigens.append(np.array([np.sqrt(eg) for s in np_sequence_eigenvalues(x) for eg in s]))
#                 # bettis = np_sequence_betti(diracs)
#                 # eigens = np_sequence_eigenvalues(diracs)
                
#                 d_feature_bettis = d_feature_bettis + dirac_bettis
                
               
                
                dmeans = np_eigen_means(dirac_eigens)
                # features = features + [means]
                # features = features + means
                d_feature_means = d_feature_means + [dmeans]

                dsums = np_eigen_sum(dirac_eigens)
                # features = features + [sums]
                # features = features + sums
                d_feature_sums = d_feature_sums + [dsums]

                dmins = np_eigen_min(dirac_eigens)
                # features = features + [mins]
                # features = features + mins
                d_feature_mins = d_feature_mins + [dmins]

                dmaxes = np_eigen_max(dirac_eigens)
                # features = features + [maxes]
                # features = features + maxes
                d_feature_maxs = d_feature_maxs + [dmaxes]

                dstds = np_eigen_std(dirac_eigens)
                # features = features + [stds]
                # features = features + stds
                d_feature_stds = d_feature_stds + [dstds]

                dgen_means = np_eigen_positive_generalized_mean(dirac_eigens)
                # features = features + [gen_means]
                # features = features + gen_means
                d_feature_gen_means = d_feature_gen_means + [dgen_means]

                dsec_moms = dirac_zeta(-2, dirac_eigens)
                # features = features + [sec_moms]
                # features = features + sec_moms
                d_feature_sec_moms = d_feature_sec_moms + [dsec_moms]

                dzet2s = dirac_zeta(2, dirac_eigens)
                # features = features + [zet2s]
                # features = features + zet2s
                d_feature_zets2 = d_feature_zets2 +[dzet2s]

                dzet1s = dirac_zeta(1, dirac_eigens)
                # features = features + [zet1s]
                # features = features + zet1s
                d_feature_zets1 = d_feature_zets1 + [dzet1s]

# #                 c1 = s_dirac_complexity(1, dirac_eigens)
# #                 # features = features + [c1]
# #                 # features = features + c1
# #                 d_feature_c1 = d_feature_c1 + [c1]
                
# #                 EPs = np_dirac_signless_Euler_Poincare(subchain_bases_dims, bettis)
# #                 # features = features + [EPs]
# #                 # features = features + EPs
# #                 d_feature_EPs = d_feature_EPs + [EPs]

# #                 stns = spanning_tree_number(c1, EPs)
# #                 # features = features + [stns]
# #                 # features = features + stns
# #                 d_feature_stns = d_feature_stns + [stns]

                dQWs = Quasi_Wiener(dirac_eigens)
                # features = features + [QWs]
                # features = features + QWs
                d_feature_QWs = d_feature_QWs + [dQWs]
                
        
        filts_bettis = filts_bettis + [feature_bettis]
        filts_means = filts_means + [feature_means]
        filts_sums = filts_sums + [feature_sums]
        filts_mins = filts_mins + [feature_mins]
        filts_maxs = filts_maxs + [feature_maxs]
        filts_stds = filts_stds + [feature_stds]
        filts_gen_means = filts_gen_means + [feature_gen_means]
        filts_sec_moms = filts_sec_moms + [feature_sec_moms]
        filts_zets2 = filts_zets2 + [feature_zets2]
        filts_zets1 = filts_zets1 + [feature_zets1]
        filts_c1 = filts_c1 + [feature_c1]
        filts_EPs = filts_EPs + [feature_EPs]
        filts_stns = filts_stns + [feature_stns]
        filts_QWs = filts_QWs + [feature_QWs]
        
        d_filts_bettis = d_filts_bettis + [d_feature_bettis]
        d_filts_means = d_filts_means + [d_feature_means]
        d_filts_sums = d_filts_sums + [d_feature_sums]
        d_filts_mins = d_filts_mins + [d_feature_mins]
        d_filts_maxs = d_filts_maxs + [d_feature_maxs]
        d_filts_stds = d_filts_stds + [d_feature_stds]
        d_filts_gen_means = d_filts_gen_means + [d_feature_gen_means]
        d_filts_sec_moms = d_filts_sec_moms + [d_feature_sec_moms]
        d_filts_zets2 = d_filts_zets2 + [d_feature_zets2]
        d_filts_zets1 = d_filts_zets1 + [d_feature_zets1]
        # d_filts_c1 = []
        # d_filts_EPs = []
        # d_filts_stns = []
        d_filts_QWs = d_filts_QWs + [d_feature_QWs]
        
        

      
    
    # return None
    # return [radii, filts_bettis,\
    #         filts_means, filts_sums, filts_mins,\
    #         filts_maxs, filts_stds, filts_gen_means,\
    #         filts_sec_moms, filts_zets2, filts_zets1,\
    #         filts_c1, filts_EPs, filts_stns, filts_QWs, d_filts_bettis]
    return [radii, filts_bettis,\
            filts_means, filts_sums, filts_mins,\
            filts_maxs, filts_stds, filts_gen_means,\
            filts_sec_moms, filts_zets2, filts_zets1,\
            filts_c1, filts_EPs, filts_stns, filts_QWs,\
            d_filts_bettis, d_filts_means, d_filts_sums,\
            d_filts_mins, d_filts_maxs, d_filts_stds,\
            d_filts_gen_means, d_filts_sec_moms, d_filts_zets2,\
            d_filts_zets1, d_filts_QWs]



def flattener(fea_seq, fil_length = 10):
    a = np.empty((fil_length,0), dtype=float)
    for fea in fea_seq:
        seqflat = []
        for y in range(len(fea)):
            flat = []
            for x in fea[y]:
                flat = flat + x
            seqflat = seqflat + [flat]
        a = np.concatenate((a, np.array(seqflat)), axis = 1)
    return a

def flattener_indivisual(fea_seq, fil_length):
    a = [[] for i in range(len(fea_seq))]

    for i, fea in zip(range(len(fea_seq)), fea_seq):
        seqflat = []
        for y in range(len(fea)):
            flat = []
            for x in fea[y]:
                flat = flat + x
            seqflat = seqflat + [flat]
        a[i] = np.array(seqflat)
    return a


def weights(basis, points):
    wgh = []
    for i in range(len(basis)):
        ds = []
        for x in basis[i]:
            buffer = []
            for y in x:
                buffer.append(points[y])
            ds.append(np.mean(np.triu(distance.cdist(np.array(buffer), np.array(buffer), 'euclidean'), 0)))
        wgh.append(np.diag(np.array(ds))**2)
    wgh[0] = np.eye(len(basis[0]))
    return wgh

def weighted_chain(chain, wghts):
    for i in range(1, len(wghts)):
        chain[i] = wghts[i-1]@chain[i]@np.linalg.inv(wghts[i])
    return chain



