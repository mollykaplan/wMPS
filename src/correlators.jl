################################################################################################
#Daubechies wavelet values
################################################################################################

#values of the N=6 Daubechies scaling function s(x) on [0, 5], in steps of dx
const sxdata=[0.,0.0477135636885,0.101396505763,0.158604211489,0.215399070084,0.284716624233,0.337127934667,0.394080128947,0.457836727203,0.530679681933,0.605178468388,0.660144859778,0.716586371058,0.775155036186,0.837684705982,0.898113049518,0.973137367421,1.05016315871,1.12798099254,1.20528197587,1.28633506943,1.22948248334,1.15723970541,1.07711397914,1.00074725511,0.889916048133,0.82991718245,0.758831434055,0.670055664191,0.556984446439,0.441122481462,0.386695807121,0.330364448733,0.2700926328,0.200333452582,0.139418881901,0.037237256857,-0.06845927815,-0.1740327981,-0.275556401991,-0.385836961046,-0.366179422596,-0.333668189291,-0.296157303242,-0.264946951462,-0.202979934521,-0.189570037759,-0.167645406296,-0.13097774463,-0.0734424884942,-0.0149705913866,-0.0211077543959,-0.0278821949469,-0.033098796308,-0.0311199431713,-0.040567571439,-0.0112874414092,0.0190279678607,0.0467214663125,0.0678657092793,0.0952675460038,0.0856559726186,0.0726300715015,0.0591546552953,0.0486546894467,0.0299440598983,0.0234756216905,0.0150748268266,0.00304042006306,-0.0141882578704,-0.0315413029749,-0.0258525194967,-0.0190756083477,-0.0121016183393,-0.00690046314216,0.00302513129194,0.000912474753948,-0.000731903906359,-0.000669676298295,0.0024087161316,0.0042343456164,0.00332740295281,0.00240190662056,0.00128445731713,0.000145936821122,-0.00159679774383,-0.000950701048111,-0.000340983533211,0.0000449331724483,-0.0000333820070649,0.000210944511631,0.000119606992823,6.98350365964*10^(-6),-0.0000472543387936,2.24774952535*10^(-6),0.0000105087281527,3.42377605237*10^(-7),5.54896037911*10^(-8),1.55451934543*10^(-8),7.12088762269*10^(-10),0.]
const sint=[sxdata[21],sxdata[41],sxdata[61],sxdata[81]] #[s(1), s(2), s(3), s(4)]
const dx=0.05 #step size between values

#maximum number of bosons per site of the MPS ψ
state_cutoff(ψ) = dim(physicalspace(ψ, 1)) - 1

check_dx(dx) = dx ≈ wMPS.dx || throw(ArgumentError("dx must be the spacing of the tabulated scaling function, wMPS.dx = $(wMPS.dx) (got $dx)"))

################################################################################################
#Correlation functions
################################################################################################

"""
Computing ⟨ψ†(x)ψ(0)⟩.

Inputs
x = some postition
ψ = MPS, in the basis of N=6 Daubechies wavelets (the scaling function is tabulated for D6 only)
dx = step-size between chosen x values for s(x), must be wMPS.dx
r = resolution

Output: the value of the correlator at x
"""
function correlator_x(x, ψ, dx, r)
    check_dx(dx)
    cutoff = state_cutoff(ψ)
    op = a_plus(; cutoff) ⊗ a_min(; cutoff)
    x_tot=0.0
    Δ=2^r
    for i=convert(Int64,ceil(Δ*x)-5):convert(Int64,floor(Δ*x))
        sx_toind=convert(Int64,round((Δ*x-i)/dx)) + 1
        for j=-4:-1
            exp_op=real(expectation_value(ψ, (i,j) => op))
            exp_v=Δ*sint[-j]*sxdata[sx_toind]*exp_op
            x_tot += exp_v
        end
    end
    return x_tot
end

"""
Computing ⟨ψ†(x)ψ†(0)ψ(x)ψ(0)⟩.

Inputs
x = some postition
ψ = MPS, in the basis of N=6 Daubechies wavelets (the scaling function is tabulated for D6 only)
dx = step-size between chosen x values for s(x), must be wMPS.dx
r = resolution

Output: the value of the correlator at x
"""
function density_correlator(x, ψ, dx, r)
    check_dx(dx)
    cutoff = state_cutoff(ψ)
    op = a_plus(; cutoff) ⊗ a_plus(; cutoff) ⊗ a_min(; cutoff) ⊗ a_min(; cutoff)
    x_tot=0.0
    Δ=2^r
    lstart=convert(Int64,ceil(Δ*x)-5)
    lend=convert(Int64,floor(Δ*x))
    for i=lstart:lend
        sx_toi=convert(Int64,round((Δ*x-i)/dx)) + 1
        for j=lstart:lend
            sx_toj=convert(Int64,round((Δ*x-j)/dx)) + 1
            for k=-4:-1, l=-4:-1
                exp_op=real(expectation_value(ψ, (i,k,j,l) => op))
                exp_v=Δ^2*sxdata[sx_toi]*sxdata[sx_toj]*sint[-l]*sint[-k]*exp_op
                x_tot += exp_v
            end
        end
    end
    return x_tot
end