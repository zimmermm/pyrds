Python implementation of a finite volume discretization to solve the reaction-diffusion-equation on a non-uniforme 1d domain.

Installation
~~~~~~~~~~~~

.. code-block:: python

   pip install pyrds


Usage
~~~~~

.. code-block:: python

	import numpy
	from collections import deque
	from pyrds import FiniteVolumeDiscretization, lapack_tridiag
	from matplotlib import pyplot


	if __name__ == '__main__':
	    ''' discrete representation of a lake '''
	    # bathymetry/morphology of a lake
	    bathymetry_z = numpy.linspace(0, 16, 9, endpoint=True)  # [m]
	    bathymetry_areas = numpy.array([494199, 440930, 393912, 348221, 301994, 240935, 153063, 69403, 18691])  # [m2]

	    # finite volume discretization
	    n_volumes = 499
	    lake = FiniteVolumeDiscretization(0, 16, n_volumes, bathymetry_z, bathymetry_areas)

	    # define time grid with a constant time step
	    dt = 1./8.  # [d]
	    tgrid = numpy.arange(0, 365+dt/2., dt).tolist()

	    ''' initial conditions '''
	    n_vars = 2
	    state_vars = numpy.hstack((
	                                numpy.zeros((n_volumes, 1)),  # [umol m-3]
	                                numpy.zeros((n_volumes, 1))   # [umol m-3]
	                             )).reshape((n_volumes*n_vars, 1))

	    ''' define constant source/sink as well as diffusivity '''
	    D = numpy.ones((n_vars, n_volumes+1)) * 0.3
	    S = numpy.zeros((n_vars, n_volumes))  # volume sources
	    F = numpy.zeros((n_vars, n_volumes+1))  # flux sources
	    # Flux of substance 0 from atmosphere
	    F[0,0] = 20
	    # Flux of substance 1 from sediment
	    F[1,-1] = -20*bathymetry_areas[0]/bathymetry_areas[-1]

	    ''' precondition the discretization system '''
	    lake.precondition(n_vars, dt)

	    ''' initialize record '''
	    state_record = deque()

	    ''' run simulation'''
	    for t in tgrid:
	        state_vars = lapack_tridiag(*lake.assembleLES_IE(state_vars, D, S, F))
	        state_record.append((state_vars, t))

	    ''' plot result '''
	    record_array = numpy.array([r[:,0] for (r, t) in state_record])
	    record_array = record_array.reshape((-1, n_vars, n_volumes))

	    substance_A_record = record_array[:,0,:] / 1000. # [uM]
	    substance_B_record = record_array[:,1,:] / 1000. # [uM]

	    def colormesh(fig, ax, tmesh, zmesh, var, title):
	        ax.invert_yaxis()
	        ax.set_title(title)
	        cbar = ax.imshow(numpy.fliplr(numpy.rot90(var, -1)), vmin=0, aspect='auto', extent=[tgrid[0], tgrid[-1], 0, 16])
	        ax.set_ylabel('Depth [m]')
	        ax.set_xlabel('Time [d]')
	        fig.colorbar(cbar, ax=ax)

	    fig = pyplot.figure()
	    ax = fig.add_subplot(211)
	    colormesh(fig, ax, tgrid, lake.grid.z_centres, substance_A_record, 'Substance A [uM]')
	    ax = fig.add_subplot(212)
	    colormesh(fig, ax, tgrid, lake.grid.z_centres, substance_B_record, 'Substance B [uM]')
	    pyplot.show()