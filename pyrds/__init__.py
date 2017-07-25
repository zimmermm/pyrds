
import numpy
from scipy import interpolate
from scipy import linalg
from scipy import sparse

"""
##########################################################
        FINITE VOLUME DISCRETIZATION
##########################################################
"""
class StaggeredGrid(object):
    def __init__(self, z_0, z_end, nz, z_areas, areas):
        ''' initialize and equally spaced staggered grid with nz volume centres and nz+1 volume faces '''
        self.z_0 = z_0
        self.z_end = z_end
        self.nz = nz
        self.z_faces = numpy.linspace(z_0, z_end, nz+1, endpoint=True)   # Positions of the volume faces
        self.z_centres = (self.z_faces[:-1]+self.z_faces[1:])/2.                   # position of the volume centres
        self.h_centres = numpy.diff(self.z_faces)                             # distance between volume centres
        self.h_faces = numpy.diff(self.z_centres)                             # height of the volumes (distance between volume faces)
        self.a_faces = interpolate.griddata(z_areas, areas, self.z_faces)   # Interpolated areas at the positions of the volume faces
        self.a_centres = (self.a_faces[:-1]+self.a_faces[1:])/2.                   # At the volume centres, the area is assumed to be the average of the face areas
        self.volumes = self.a_centres*self.h_centres

    def unpack(self):
        return (self.z_0, self.z_end, self.nz,
                self.z_centres, self.z_faces,
                self.h_centres, self.h_faces,
                self.a_centres, self.a_faces)

class FiniteVolumeDiscretization(object):
    def __init__(self, z_0, z_end, nz, z_areas, areas):
        ''' initialize and equally spaced staggered grid with nz volume centres and nz+1 volume faces '''
        self.grid = StaggeredGrid(z_0, z_end, nz, z_areas, areas)


    def precondition(self, n_vars, dt):
        ''' Precondition the discretization scheme '''
        # unpacking some variables for shorter notation
        z_0, z_end, nz, z_centres, z_faces, h_centres, h_faces, a_centres, a_faces = self.grid.unpack()

        # constants used to build the tridiagonals
        form = -4.*dt*a_faces[1:-1]/(h_centres[1:]+h_centres[:-1])
        form1 = numpy.hstack(([0],                                      # no-flux boundary
                              form/(a_faces[2:]+a_faces[1:-1])/h_centres[1:]))
        form2 = numpy.hstack((form/(a_faces[1:-1]+a_faces[:-2])/h_centres[:-1],
                              [0]))                                     # no-flux boundary
        self.form1 = numpy.tile(form1, n_vars)
        self.form2 = numpy.tile(form2, n_vars)
        self.dim = nz*n_vars
        self.dt = dt


    def assembleLineareEquationSystemCore(self, diffusivities, sources, fluxes):
        # get the diffusivities
        D1 = diffusivities[:, :-1].ravel()
        D2 = diffusivities[:, 1:].ravel()
        # get source and sink rates
        # extend in volume sources and area sources? => flux_term which
        # returns fluxes at the top and bottom of volumes
        S = sources.reshape((self.dim, 1))
        f = [flux*self.grid.a_faces for flux in fluxes]
        Q = numpy.array([((flux[:-1]-flux[1:])/self.grid.volumes) for flux in f]).reshape((self.dim, 1))

        # upper and lower diagonals are both the same for implicit euler
        # and crank-nicolson
        ld = self.form1*D1
        ud = self.form2*D2
        # right hand side
        rhs = self.dt*(S+Q)

        return (ld, ud, rhs)


    def assembleLES_IE(self, state_vars, diffusivities, sources, fluxes):
        ld, ud, rhs = self.assembleLineareEquationSystemCore(diffusivities, sources, fluxes)
        md = (1. - ld - ud)
        # reform diagonal to fit scipy linalg convention
        ld = numpy.hstack((ld[1:], [0]))
        ud = numpy.hstack(([0], ud[:-1]))
        return (ld, md, ud, rhs+state_vars.clip(0))


    def assembleLES_CN(self, state_vars, diffusivities, sources, fluxes):
        A_ld, A_ud, rhs = self.assembleLineareEquationSystemCore(diffusivities, sources, fluxes)

        # IMPLICIT PART of the crank-nicolson stencil:
        A_ld = 0.5*A_ld
        A_ud = 0.5*A_ud
        A_md = (1. - A_ld - A_ud)
        # reform upper and lower diagonal to fit scipy convention
        A_ld = numpy.hstack((A_ld[1:], [0]))
        A_ud = numpy.hstack(([0], A_ud[:-1]))

        # EXPLICIT Contribution of the crank-nicolson stencil
        B_md = 2. - A_md
        B = sparse.spdiags([-A_ld, B_md, -A_ud], [-1, 0, 1], self.dim, self.dim)
        return (A_ld, A_md, A_ud, rhs+B.dot(state_vars))


"""
##########################################################
        SOLVER for Linear Equation System
##########################################################
"""


def lapack_tridiag(ld, md, ud, rhs):
    '''
    Solve tridiagonal system with scipy.linalg.solve_banded
    solve_banded uses LAPACK routines to solve the linear equation system
    '''
    return linalg.solve_banded((1, 1), numpy.vstack((ud, md, ld)), rhs, overwrite_ab=True, overwrite_b=True)


