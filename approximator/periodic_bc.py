import numpy as np
from pysph.base.utils import get_particle_array
from pysph.tools.interpolator import get_bounding_box, Interpolator
from pysph.sph.equation import Equation, Group
from compyle.api import declare

def get_max_h_in_arrays(particle_arrays):
    hmax = -1.0
    for array in particle_arrays:
        hmax = max(array.h.max(), hmax)
    return hmax

def create_ghosts(particle_arrays, dim, dx, num_layers=2):
    concat = np.concatenate
    if num_layers <= 0:
        raise ValueError("num_layers must be > 0")
    Ndx = num_layers*dx
    hmax = get_max_h_in_arrays(particle_arrays)
    
    tight_bounds = get_bounding_box(particle_arrays, tight=True)
    xmin, xmax, ymin, ymax, zmin, zmax = tight_bounds
    Lx, Ly, Lz = xmax-xmin, ymax-ymin, zmax-zmin

    # FIXME: Implement for 2D and 3D
    if dim != 1:
        raise NotImplementedError("Only 1D implemented so far.")

    # Create Ghost-2
    _xmin = xmin + dx/2
    _xmax = xmax - dx/2
    if dim == 1:
        x1 = np.arange(_xmin, _xmin+Ndx, dx)
        x2 = np.arange(_xmax, _xmax-Ndx, -dx)
        if len(x1) > num_layers: x1 = x1[:-1]
        if len(x2) > num_layers: x2 = x2[:-1]
        
        x = concat([x1, x2])
        y, z = np.zeros_like(x), np.zeros_like(x)
        
        xr, yr, zr = x.ravel(), y.ravel(), z.ravel()
        gid = np.arange(len(xr))
    

    ghost2 = get_particle_array(
        name="ghost2", x=xr, y=yr, z=zr, h=hmax, gid=gid, m=1.
    )
    ghost2.ensure_properties(particle_arrays[0])

    # Create Ghost-1
    _Lx = Lx + dx
    if dim == 1:
        # Get east particles
        mask = (xr >= _xmin) & (xr <= _xmin+Ndx)
        x = xr[mask] + _Lx
        g2id = gid[mask]

        # Get west particles
        mask = (xr >= _xmax-Ndx) & (xr <= _xmax)
        x = concat([x, xr[mask] - _Lx])
        g2id = concat([g2id, gid[mask]])
        y, z = np.zeros_like(x), np.zeros_like(x)

        xr, yr, zr = x.ravel(), y.ravel(), z.ravel()
        gid = np.arange(len(xr))

    ghost1 = get_particle_array(
        name="ghost1", x=xr, y=yr, z=zr, h=hmax, gid=gid, m=1.
    )
    ghost1.add_property(
        **{'name': 'g2id', 'data': g2id, 'type': 'unsigned int'}
    )
    ghost1.ensure_properties(particle_arrays[0])

    return ghost1, ghost2

def pre_step_interpolate(
    particle_arrays, ghost2, dim, method='order1', interp_ob=None
):    
    # Get the properties to interpolate
    props = list(ghost2.get_property_arrays().keys())
    props_to_ignore = ['pid', 'gid', 'tag', 'g2id', 'm', 'h', 'x', 'y', 'z']
    props_to_interpolate = sorted([p for p in props if p not in props_to_ignore])
    
    # Setup interpolator
    if interp_ob is None:
        print(f"\nInterpolating: {props_to_interpolate}")

        x, y, z = ghost2.x, ghost2.y, ghost2.z
        interp_ob = Interpolator(
            particle_arrays=particle_arrays,
            x=x, y=y, z=z,
            method=method,
        )
    
    # Update positions
    else:
        interp_ob.update_particle_arrays(particle_arrays)

    # Interpolate    
    for prop in props_to_interpolate:
        if prop in ghost2.stride:
            data = interp_ob.interpolate_prop_with_stride(prop)
        else:
            data = interp_ob.interpolate(prop)

        # Set the interpolated values to the ghost particles
        ghost2.get(prop)[:] = data
    return interp_ob, props_to_interpolate

def pre_step_copy_props_to_ghost1(ghost1, ghost2, props_to_interpolate):
    for i in ghost1.gid:
        g2id = ghost1.g2id[i]
        for prop in props_to_interpolate:
            if not (prop in ghost1.stride):
                ghost1.get(prop)[i] = ghost2.get(prop)[g2id]
            else:
                st = ghost1.stride[prop]
                for j in range(st):
                    ghost1.get(prop)[i*st+j] = ghost2.get(prop)[g2id*st+j]
                