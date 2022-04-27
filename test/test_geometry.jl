# Unit tests for JUDI Geometry structure
# Philipp Witte (pwitte.slim@gmail.com)
# May 2018
#
# Mathias Louboutin, mlouboutin3@gatech.edu
# Updated July 2020

datapath = joinpath(dirname(pathof(JUDI)))*"/../data/"

@testset "Geometry Unit Test with $(nsrc) sources" for nsrc=[1, 2]
    @timeit TIMEROUTPUT "Geometry (nsrc=$(nsrc))" begin
        # Constructor if nt is not passed
        xsrc = convertToCell(range(100f0, stop=1100f0, length=2)[1:nsrc])
        ysrc = convertToCell(range(0f0, stop=0f0, length=nsrc))
        zsrc = convertToCell(range(20f0, stop=20f0, length=nsrc))

        geometry =  Geometry(xsrc, ysrc, zsrc; dt=2f0, t=1000f0)

        @test isequal(typeof(geometry.xloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.yloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.zloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.nt), Array{Integer, 1})
        @test isequal(typeof(geometry.dt), Array{Float32, 1})
        @test isequal(typeof(geometry.t), Array{Float32, 1})

        # Constructor if coordinates are not passed as a cell arrays
        xsrc = range(100f0, stop=1100f0, length=2)[1:nsrc]
        ysrc = range(0f0, stop=0f0, length=nsrc)
        zsrc = range(20f0, stop=20f0, length=nsrc)

        geometry = Geometry(xsrc, ysrc, zsrc; dt=4f0, t=1000f0, nsrc=nsrc)

        @test isequal(typeof(geometry.xloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.yloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.zloc), Array{Array{Float32,1}, 1})
        @test isequal(typeof(geometry.nt), Array{Integer, 1})
        @test isequal(typeof(geometry.dt), Array{Float32, 1})
        @test isequal(typeof(geometry.t), Array{Float32, 1})

        # Set up source geometry object from in-core data container
        block = segy_read(datapath*"unit_test_shot_records_$(nsrc).segy"; warn_user=false)
        src_geometry = Geometry(block; key="source", segy_depth_key="SourceSurfaceElevation")
        rec_geometry = Geometry(block; key="receiver", segy_depth_key="RecGroupElevation")

        @test isequal(typeof(src_geometry), GeometryIC{Float32})
        @test isequal(typeof(rec_geometry), GeometryIC{Float32})
        @test isequal(get_header(block, "SourceSurfaceElevation")[1], src_geometry.zloc[1][1])
        @test isequal(get_header(block, "RecGroupElevation")[1], rec_geometry.zloc[1][1])
        @test isequal(get_header(block, "SourceX")[1], src_geometry.xloc[1][1])
        @test isequal(get_header(block, "GroupX")[1], rec_geometry.xloc[1][1])

        # Set up geometry summary from out-of-core data container
        container = segy_scan(datapath, "unit_test_shot_records_$(nsrc)", ["GroupX", "GroupY", "RecGroupElevation", "SourceSurfaceElevation", "dt"])
        src_geometry = Geometry(container; key="source", segy_depth_key="SourceSurfaceElevation")
        rec_geometry = Geometry(container; key="receiver", segy_depth_key="RecGroupElevation")

        @test isequal(typeof(src_geometry), GeometryOOC{Float32})
        @test isequal(typeof(rec_geometry), GeometryOOC{Float32})
        @test isequal(src_geometry.key, "source")
        @test isequal(rec_geometry.key, "receiver")
        @test isequal(src_geometry.segy_depth_key, "SourceSurfaceElevation")
        @test isequal(rec_geometry.segy_depth_key, "RecGroupElevation")
        @test isequal(prod(size(block.data)), sum(rec_geometry.nrec .* rec_geometry.nt))

        # Set up geometry summary from out-of-core data container passed as cell array
        container_cell = Array{SegyIO.SeisCon}(undef, nsrc)
        for j=1:nsrc
            container_cell[j] = split(container, j)
        end

        src_geometry = Geometry(container_cell; key="source", segy_depth_key="SourceSurfaceElevation")
        rec_geometry = Geometry(container_cell; key="receiver", segy_depth_key="RecGroupElevation")

        @test isequal(typeof(src_geometry), GeometryOOC{Float32})
        @test isequal(typeof(rec_geometry), GeometryOOC{Float32})
        @test isequal(src_geometry.key, "source")
        @test isequal(rec_geometry.key, "receiver")
        @test isequal(src_geometry.segy_depth_key, "SourceSurfaceElevation")
        @test isequal(rec_geometry.segy_depth_key, "RecGroupElevation")
        @test isequal(prod(size(block.data)), sum(rec_geometry.nrec .* rec_geometry.nt))

        # Load geometry from out-of-core Geometry container
        src_geometry_ic = Geometry(src_geometry)
        rec_geometry_ic = Geometry(rec_geometry)

        @test isequal(typeof(src_geometry_ic), GeometryIC{Float32})
        @test isequal(typeof(rec_geometry_ic), GeometryIC{Float32})
        @test isequal(get_header(block, "SourceSurfaceElevation")[1], src_geometry_ic.zloc[1][1])
        @test isequal(get_header(block, "RecGroupElevation")[1], rec_geometry_ic.zloc[1][1])
        @test isequal(get_header(block, "SourceX")[1], src_geometry_ic.xloc[1][1])
        @test isequal(get_header(block, "GroupX")[1], rec_geometry_ic.xloc[1][1])

        # Subsample in-core geometry structure
        src_geometry_sub = subsample(src_geometry_ic, 1)
        @test isequal(typeof(src_geometry_sub), GeometryIC{Float32})
        @test isequal(length(src_geometry_sub.xloc), 1)
        src_geometry_sub = subsample(src_geometry_ic, 1:1)
        @test isequal(typeof(src_geometry_sub), GeometryIC{Float32})
        @test isequal(length(src_geometry_sub.xloc), 1)

        inds = nsrc > 1 ? (1:nsrc) : 1
        src_geometry_sub = subsample(src_geometry_ic, inds)
        @test isequal(typeof(src_geometry_sub), GeometryIC{Float32})
        @test isequal(length(src_geometry_sub.xloc), nsrc)

        # Subsample out-of-core geometry structure
        src_geometry_sub = subsample(src_geometry, 1)
        @test isequal(typeof(src_geometry_sub), GeometryOOC{Float32})
        @test isequal(length(src_geometry_sub.dt), 1)
        @test isequal(src_geometry_sub.segy_depth_key, "SourceSurfaceElevation")

        src_geometry_sub = subsample(src_geometry, inds)
        @test isequal(typeof(src_geometry_sub), GeometryOOC{Float32})
        @test isequal(length(src_geometry_sub.dt), nsrc)
        @test isequal(src_geometry_sub.segy_depth_key, "SourceSurfaceElevation")

        # Compare if geometries match
        @test compareGeometry(src_geometry_ic, src_geometry_ic)
        @test compareGeometry(rec_geometry_ic, rec_geometry_ic)

        @test compareGeometry(src_geometry, src_geometry)
        @test compareGeometry(rec_geometry, rec_geometry)

        # Check if 'limit_model_to_receiver_area' works with nonzero origin
        # Set up model structure
        n = (100, 100, 100)    # (x,y,z) or (x,z)
        d = (10., 10., 10.)
        o = (100., 100., 0.)    # set nonzero origin

        # Velocity [km/s] (ones same as [s^2/km^2])
        m = ones(Float32,n)

        # Setup model structure
        model = Model(n, d, o, m)

        # Set up 3D receiver geometry by defining one receiver vector in each x and y direction
        nxrec = 5
        nyrec = 3
        xrec = range(300f0, stop=700f0, length=nxrec)
        yrec = range(400f0, stop=600f0, length=nyrec)
        zrec = 50f0

        # Construct 3D grid from basis vectors
        (xrec, yrec, zrec) = setup_3D_grid(xrec, yrec, zrec)

        # receiver sampling and recording time
        timeR = 100f0   # receiver recording time [ms]
        dtR = 4f0    # receiver sampling interval

        # Set up receiver structure
        recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=1)

        # Set up source geometry (cell array with source locations for each shot)
        xsrc = convertToCell([500f0])
        ysrc = convertToCell([500f0])
        zsrc = convertToCell([0f0])

        # source sampling and number of time steps
        timeS = 100f0   # source length in [ms]
        dtS = 2f0    # source sampling interval

        # Set up source structure
        srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)

        buffer = 100f0
        model_out = limit_model_to_receiver_area(srcGeometry, recGeometry, model, buffer)

        # Check results
        # min/max X
        @test isequal(model_out.o[1], xrec[1]-buffer)
        @test isequal(model_out.o[1] + model_out.d[1]*(model_out.n[1]-1), xrec[end]+buffer)
        # min/max Y
        @test isequal(model_out.o[2], yrec[1]-buffer)
        @test isequal(model_out.o[2] + model_out.d[2]*(model_out.n[2]-1), yrec[end]+buffer)
    end
end
