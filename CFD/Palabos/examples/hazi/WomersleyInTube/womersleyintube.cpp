#include "palabos3D.h"
#include "palabos3D.hh"

using namespace plb;

typedef double T;
typedef Array<T, 3> Velocity;
#define DESCRIPTOR descriptors::D3Q19Descriptor

plint xDirection = 0;
plint yDirection = 1;
plint borderWidth = 1;       // Because the Guo boundary condition acts in a one-cell layer.
                             // Requirement: margin>=borderWidth.
plint margin = 1;            // Extra margin of allocated cells around the obstacle.
plint extraLayer = 0;        // Make the bounding box larger; for visualization purposes
                             //   only. For the simulation, it is OK to have extraLayer=0.
const plint blockSize = 20;  // Zero means: no sparse representation.
const plint extendedEnvelopeWidth =
    2;  // Because the Guo boundary condition needs 2-cell neighbor access.

plint n0 = 15;  // Reference resolution.
plint ny = 20;  // Resolution (cylinder diameter).
plint nz = ny;
plint nx = ny;
T radius = (T)ny / 2.;
Array<T, 3> originalCenter(0.0, 0.0, 0.0);
Array<T, 3> inletCenter(0.0, 0.0, 0.0), outletCenter(0.0, 0.0, 0.0);
T length = (T)nx*4;  // Cylinder length (another good choice: 4*nx).
plint nAxial =
    nx / 2;  // Two parameters needed for the creation of the triangularized cylinder surface.
plint nCirc = 3 * ny / 2;

T Reynolds = 10.;
T uRef = 0.02;

T uAverage = uRef * n0 / ny;

#ifndef PLB_REGRESSION
plint maxIter = 100000;  // Maximum number of iterations for the simulation.
#else
plint maxIter = 101;
#endif
plint outIter = 100;  // Number of iterations for printing the average kinetic energy on the screen.
plint saveIter = 100;  // Number of iterations for saving data in the disk.

bool useAllDirections = true;  // Extrapolation scheme for the off lattice boundary condition.
bool useRegularized = true;    // Use an off lattice boundary condition which is closer in spirit to
                               //   regularized boundary conditions.

plint particleTimeFactor = 1;  // If this variable has value 2, this means that the particles have
                               //   a time step two times bigger than the fluid.
T particleProbabilityPerCell =
    2.5e-3;  // Probability of injecting a particle into an injection cell at each time step.
T cutOffSpeedSqr = 1.e-8;  // Criterion to eliminate particles with very small velocity.

// This function object is a domain functional which is used for injecting particles from
//   an area close to the inlet of the tube.
class CircularInjection {
public:
    CircularInjection(T radius_, Array<T, 3> center_) : radius(radius_), center(center_) { }
    bool operator()(Array<T, 3> const &pos) const
    {
        return (util::sqr(pos[1] - center[1]) + util::sqr(pos[2] - center[2])) < util::sqr(radius);
    }

private:
    T radius;
    Array<T, 3> center;
};

T womersleyForce(T t, T A, T omega)
{
    return A * std::cos(omega * t);
}

void iniLattice(MultiBlockLattice3D<T, DESCRIPTOR> &lattice, VoxelizedDomain3D<T> &voxelizedDomain)
{
    // Switch all remaining outer cells to no-dynamics, except the outer
    //   boundary layer, and keep the rest as BGKdynamics.
    defineDynamics(
        lattice, voxelizedDomain.getVoxelMatrix(), lattice.getBoundingBox(),
        new NoDynamics<T, DESCRIPTOR>, voxelFlag::outside);
    initializeAtEquilibrium(
        lattice, lattice.getBoundingBox(), (T)1., Array<T, 3>((T)0., (T)0., (T)0.));
    lattice.initialize();
}

int main(int argc, char *argv[])
{
    plbInit(&argc, &argv);
    global::directories().setOutputDir("./output/");

    uint32_t seed = 1;

    // Create the cylinder surface as a set of triangles.
    TriangleSet<T> triangleSet;
    triangleSet = constructCylinder<T>(originalCenter, radius, radius, length, nAxial, nCirc);

    // The next few lines of code are typical. They transform the surface geometry of the
    //   tube to more efficient data structures that are internally used by palabos.
    //   The TriangleBoundary3D structure will be later used to assign proper boundary conditions.
    DEFscaledMesh<T> defMesh(triangleSet, ny, yDirection, margin, extraLayer);
    defMesh.getMesh().inflate();
    TriangleBoundary3D<T> boundary(defMesh);


    T nu = uAverage * 2. * radius / Reynolds;
    T omega = 1. / (3. * nu + 0.5);
    pcout << "omega=" << omega << std::endl;

    boundary.getMesh().writeAsciiSTL("cylinder.stl");
    pcout << "Number of triangles: " << boundary.getMesh().getNumTriangles() << std::endl;

    // The tube simulation is an interior (as opposed to exterior) flow problem. For
    //   this reason, the lattice nodes that lie inside the computational domain must
    //   be identified and distinguished from the ones that lie outside of it. This is
    //   handled by the following voxelization process.
    pcout << std::endl << "Voxelizing the domain." << std::endl;
    const int flowType = voxelFlag::inside;
    VoxelizedDomain3D<T> voxelizedDomain(
        boundary, flowType, extraLayer, borderWidth, extendedEnvelopeWidth, blockSize);
    pcout << getMultiBlockInfo(voxelizedDomain.getVoxelMatrix()) << std::endl;

    std::unique_ptr<MultiBlockLattice3D<T, DESCRIPTOR> > lattice =
        generateMultiBlockLattice<T, DESCRIPTOR>(
            voxelizedDomain.getVoxelMatrix(), extendedEnvelopeWidth,
            new BGKdynamics<T, DESCRIPTOR>(omega));

    // The Guo off lattice boundary condition is set up.
    pcout << "Creating boundary condition." << std::endl;

    lattice->periodicity().toggle(2, true);

    BoundaryProfiles3D<T, Velocity> profiles;
    profiles.defineInletOutletTags(boundary, xDirection);
    //profiles.setInletOutlet(
    //    new PoiseuilleProfile3D<T>(uAverage), new PoiseuilleProfile3D<T>(-uAverage));
    /*profiles.setInletOutlet(
        new PoiseuilleProfile3D<T>(uAverage), new 
        DensityNeumannBoundaryProfile3D<T>);*/
    GuoOffLatticeModel3D<T, DESCRIPTOR> *model = new GuoOffLatticeModel3D<T, DESCRIPTOR>(
        new TriangleFlowShape3D<T, Array<T, 3> >(voxelizedDomain.getBoundary(), profiles), flowType,
        useAllDirections);
    model->selectUseRegularizedModel(useRegularized);
    OffLatticeBoundaryCondition3D<T, DESCRIPTOR, Velocity> boundaryCondition(
        model, voxelizedDomain, *lattice);
    boundaryCondition.insert();

    pcout << std::endl << "Initializing lattice." << std::endl;
    iniLattice(*lattice, voxelizedDomain);

    bool checkForErrors = true;
    pcout << std::endl << "Starting simulation." << std::endl;
    for (plint iT = 0; iT < parameters.nStep(maxT); ++iT){
        if (iT % outIter == 0) {
            pcout << "Iteration= " << i << "; "
                  << "Average energy: " << boundaryCondition.computeAverageEnergy() << std::endl;
            /*
            pcout << "Number of particles in the tube: "
                  << countParticles(*particles, particles->getBoundingBox()) << std::endl;
            */
        }
#ifndef PLB_REGRESSION
        if (i % saveIter == 0 && i > 0) {
            pcout << "Write visualization files." << std::endl;
            VtkImageOutput3D<T> vtkOut("volume", 1.);
            vtkOut.writeData<float>(*boundaryCondition.computePressure(), "p", 1.);
            vtkOut.writeData<float>(*boundaryCondition.computeVelocityNorm(), "u", 1.);

        }
#endif

        lattice->collideAndStream();

        if (checkForErrors) {
            abortIfErrorsOccurred();
            checkForErrors = false;
        }
    }

    //delete particles;
    return 0;
}
