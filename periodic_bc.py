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
    

    ghost2_pa = get_particle_array(
        name="ghost2", x=xr, y=yr, z=zr, h=hmax, gid=gid
    )
    ghost2_pa.ensure_properties(particle_arrays[0])

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

    ghost1_pa = get_particle_array(
        name="ghost1", x=xr, y=yr, z=zr, h=hmax, gid=gid,
    )
    ghost1_pa.add_property(
        **{'name': 'g2id', 'data': g2id, 'type': 'unsigned int'}
    )
    ghost1_pa.ensure_properties(particle_arrays[0])

    return ghost1_pa, ghost2_pa

def pre_step_interpolate(
    particle_arrays, ghost2_pa, method, dim, interp_ob=None
):
    # Setup interpolator
    if interp_ob is None:
        x, y, z = ghost2_pa.x, ghost2_pa.y, ghost2_pa.z
        interp_ob = Interpolator(
            particle_arrays=particle_arrays,
            x=x, y=y, z=z,
            method=method,
        )
    # Update positions
    else:
        interp_ob.update_particle_arrays(particle_arrays)

    # Interpolate
    # FIXME: How to interpolate over a generalised set of properties?
    # FIXME: How to interpolate non-scalar properties?
    rho = interp_ob.interpolate('rho')
    p = interp_ob.interpolate('p')
    u = interp_ob.interpolate('u')
    au = interp_ob.interpolate('au')

    # FIXME: How to set the values of the ghost particles for a generalised
    # set of properties?
    ghost2_pa.rho[:] = rho
    ghost2_pa.p[:] = p
    ghost2_pa.u[:] = u
    ghost2_pa.au[:] = au

    if dim > 1:
        v = interp_ob.interpolate('v')
        av = interp_ob.interpolate('av')
        ghost2_pa.v[:] = v
        ghost2_pa.av[:] = av
    if dim > 2:
        w = interp_ob.interpolate('w')
        aw = interp_ob.interpolate('aw')
        ghost2_pa.w[:] = w
        ghost2_pa.aw[:] = aw


    # Additional props
    f = interp_ob.interpolate('f')
    ghost2_pa.f[:] = f
    fx = interp_ob.interpolate('fx')
    ghost2_pa.fx[:] = fx

    return interp_ob


class CopyPropsToGhost1(Equation):
    def initialize(
        self, d_idx, s_idx, d_g2id, d_rho, d_p, d_u, d_au, d_v, d_av, d_w,
        d_aw, d_f, d_fx, s_rho, s_p, s_u, s_au, s_v, s_av, s_w, s_aw, s_f,
        s_fx
    ):
        idx = declare('int')
        idx = d_g2id[d_idx]

        # Copy props from ghost2 (source) to ghost1 (destination)
        d_rho[d_idx] = s_rho[idx]
        d_p[d_idx] = s_p[idx]
        d_u[d_idx] = s_u[idx]
        d_au[d_idx] = s_au[idx]
        d_v[d_idx] = s_v[idx]
        d_av[d_idx] = s_av[idx]
        d_w[d_idx] = s_w[idx]
        d_aw[d_idx] = s_aw[idx]

        d_f[d_idx] = s_f[idx]
        d_fx[d_idx] = s_fx[idx]

def pre_step_copy_props_to_ghost1(
        ghost1_pa, ghost2_pa, dim, t, dt, copy_ob=None
    ):
    if copy_ob is None:
        from pysph.tools.sph_evaluator import SPHEvaluator
        equations = [
            Group(
                equations=[
                    CopyPropsToGhost1(dest=ghost1_pa.name, sources=ghost2_pa.name)
                ],
                real=True, update_nnps=False
            )
        ]

        print(equations)
        
        copy_ob = SPHEvaluator(
            arrays=[ghost1_pa, ghost2_pa], equations=equations,
            dim=dim, kernel=None, backend='cython'
        )
    else:
        copy_ob.update()
        copy_ob.evaluate(t=t, dt=dt)

    return copy_ob
