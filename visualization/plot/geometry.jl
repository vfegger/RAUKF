using Plots
using LaTeXStrings

function generateGeometry()
    plt = plot3d(
        1,
        xlims = (0,1),
        ylims = (0,1),
        zlims = (0,2.0),
        xticks= ([0,1.0],["0","a"]),
        yticks= ([0,1.0],["0","b"]),
        zticks= ([0,1.0],["0","c"]),
        legend = false
    )
    p = [0,1,NaN]
    s0 = [0,0,NaN]
    s1 = [1,1,NaN]
    p4 = vcat(p,p,p,p)
    sA = vcat(s0,s1,s0,s1)
    sB = vcat(s0,s0,s1,s1)  
    x = vcat(p4,sA,sB)
    y = vcat(sA,sB,p4)
    z = vcat(sB,p4,sA)
    plot!(plt,x,y,z)
    savefig(plt,"Geometry.png")
end