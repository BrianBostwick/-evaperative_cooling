//! Single particle in a cross beam optical dipole trap
extern crate atomecs as lib;
extern crate nalgebra;
use lib::atom::{Atom, Force, Mass, Position, Velocity};
use lib::dipole::{self, DipolePlugin};
use lib::integrator::Timestep;
use lib::laser::{self, LaserPlugin};
use lib::laser::gaussian::GaussianBeam;
use lib::laser::intensity::{LaserIntensitySamplers};
use lib::output::file::{FileOutputPlugin, Text, XYZ};
use lib::simulation::SimulationBuilder;
use nalgebra::Vector3;
use specs::prelude::*;
use std::time::Instant;
use rand_distr::{Distribution, Normal};
use lib::initiate::NewlyCreated;
use std::fs::File;
use std::io::{Error, Write};
use lib::collisions::{CollisionPlugin, ApplyCollisionsOption, CollisionParameters, CollisionsTracker};
use lib::sim_region::{ SimulationVolume, VolumeType};
use lib::shapes::Sphere;
use lib::ramp;
use lib::ramp::{Lerp, Ramp, RampUpdateSystem};

const BEAM_NUMBER: usize = 2;

fn main() {
    let now = Instant::now();

    // Configure simulation output.
    let mut sim_builder = SimulationBuilder::default();
    sim_builder.world.register::<NewlyCreated>();

    sim_builder.world.register::<RampUpdateSystem>();

    sim_builder.add_plugin(LaserPlugin::<{BEAM_NUMBER}>);
    sim_builder.add_plugin(DipolePlugin::<{BEAM_NUMBER}>);

    sim_builder.add_end_frame_systems();
    sim_builder.add_plugin(CollisionPlugin);

    sim_builder.add_plugin(FileOutputPlugin::<Position, Text, Atom>::new("D:/data_1/pos_3.txt".to_string(), 100));
    sim_builder.add_plugin(FileOutputPlugin::<Velocity, Text, Atom>::new("D:/data_1/vel_3.txt".to_string(), 100));
    sim_builder.add_plugin(FileOutputPlugin::<Position, XYZ, Atom>::new("D:/data_1/position_3.xyz".to_string(), 100));
    sim_builder.add_plugin(FileOutputPlugin::<LaserIntensitySamplers<{BEAM_NUMBER}>, Text, LaserIntensitySamplers<{BEAM_NUMBER}>>::new("D:/data_1/intensity_3.txt".to_string(), 100));

    // sim_builder.add_plugin(FileOutputPlugin::<Position, Text, Atom>::new("pos.txt".to_string(), 1));
    // sim_builder.add_plugin(FileOutputPlugin::<Velocity, Text, Atom>::new("vel.txt".to_string(), 1));

    let mut sim = sim_builder.build();

    // Creating simulation volume
    let sphere_pos = Vector3::new(0.0, 0.0, 0.0);
    let sphere_radius = 60.0e-6 / 2.0_f64.sqrt();
    sim.world
        .create_entity()
        .with(Position { pos: sphere_pos })
        .with(Sphere {
            radius: sphere_radius,
        })
        .with(SimulationVolume {
            volume_type: VolumeType::Inclusive,
        })
        .build();

    // Create dipole laser.
    let power = 7.0;
    let e_radius = 60.0e-6 / 2.0_f64.sqrt();
    let wavelength = 1064.0e-9;

    let gaussian_beam_one = GaussianBeam {
        intersection: Vector3::new(0.0, 0.0, 0.0),
        e_radius,
        power,
        direction: Vector3::x(),
        rayleigh_range: crate::laser::gaussian::calculate_rayleigh_range(&wavelength, &e_radius),
        ellipticity: 0.0,
    };

    let gaussian_beam_two = GaussianBeam {
        intersection: Vector3::new(0.0, 0.0, 0.0),
        e_radius,
        power,
        direction: Vector3::y(),
        rayleigh_range: crate::laser::gaussian::calculate_rayleigh_range(&wavelength, &e_radius),
        ellipticity: 0.0,
    };

    sim.world
        .create_entity()
        .with(gaussian_beam_one)
        .with(dipole::DipoleLight { wavelength })
        .with(laser::frame::Frame {
            x_vector: Vector3::y(),
            y_vector: Vector3::z(),
        })
        .build();

    sim.world
        .create_entity()
        .with(gaussian_beam_two)
        .with(dipole::DipoleLight { wavelength })
        .with(laser::frame::Frame {
            x_vector: Vector3::x(),
            y_vector: Vector3::z(),
        })
        .build();

    let p_dist = Normal::new(0.0, 50e-6).unwrap();
    let v_dist = Normal::new(0.0, 0.004).unwrap(); // ~100nK

    // Create a single test atom
    let atom_number = 25000;
    for _ in 0..atom_number {
        sim.world
            .create_entity()
            .with(Atom)
            .with(Mass { value: 87.0 })
            .with(Force::new())
            .with(Position {
                pos: Vector3::new(
                    p_dist.sample(&mut rand::thread_rng()),
                    p_dist.sample(&mut rand::thread_rng()),
                    p_dist.sample(&mut rand::thread_rng()),
                ),
            })
            .with(Velocity {
                vel: Vector3::new(
                    v_dist.sample(&mut rand::thread_rng()),
                    v_dist.sample(&mut rand::thread_rng()),
                    v_dist.sample(&mut rand::thread_rng()),
                ),
            })
            .with(dipole::Polarizability::calculate_for(
                wavelength, 461e-9, 32.0e6,
            ))
            .with(lib::initiate::NewlyCreated)
            .build();
    }

    sim.world.insert(ApplyCollisionsOption);
    sim.world.insert(CollisionParameters {
        macroparticle: 4e2,
        box_number: 1000,  //Any number large enough to cover entire cloud with collision boxes. Overestimating box number will not affect performance.
        box_width: 1e-6, //Too few particles per box will both underestimate collision rate and cause large statistical fluctuations.
                          //Boxes must also be smaller than typical length scale of density variations within the cloud, since the collisions model treats gas within a box as homogeneous.
        sigma: 1.95e-19,   //Approximate collisional cross section of Sr
        collision_limit: 10_000.0, //Maximum number of collisions that can be calculated in one frame.
                                       //This avoids absurdly high collision numbers if many atoms are initialised with the same position, for example.
    });
    sim.world.insert(CollisionsTracker {
        num_collisions: Vec::new(),
        num_atoms: Vec::new(),
        num_particles: Vec::new(),
    });

    // Define timestep
    sim.world.insert(Timestep { delta: 1.0e-6 });
    //Timestep must also be much smaller than mean collision time

    let mut filename = File::create("D:/data_1/collisions_3.txt").expect("Cannot create file.");

    // Run the simulation for a number of steps.
    for _i in 0..50_000 {
        sim.step();

        if (_i > 0) && (_i % 50_i32 == 0) {
            let tracker = sim.world.read_resource::<CollisionsTracker>();
            let _result = write_collisions_tracker(
                &mut filename,
                &_i,
                &tracker.num_collisions,
                &tracker.num_atoms,
                &tracker.num_particles,
            )
            .expect("Could not write collision stats file.");
        }
    }
    println!("Simulation completed in {} ms.", now.elapsed().as_millis());
}


// Write collision stats to file

fn write_collisions_tracker(
    filename: &mut File,
    step: &i32,
    num_collisions: &Vec<i32>,
    num_atoms: &Vec<f64>,
    num_particles: &Vec<i32>,
) -> Result<(), Error> {
    let str_collisions: Vec<String> = num_collisions.iter().map(|n| n.to_string()).collect();
    let str_atoms: Vec<String> = num_atoms.iter().map(|n| format!("{:.2}", n)).collect();
    let str_particles: Vec<String> = num_particles.iter().map(|n| n.to_string()).collect();
    write!(
        filename,
        "{:?}\r\n{:}\r\n{:}\r\n{:}\r\n",
        step,
        str_collisions.join(" "),
        str_atoms.join(" "),
        str_particles.join(" ")
    )?;
    Ok(())
}
