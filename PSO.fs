//--------------------------------------------------------------------------------------------------------------
//
//  CSCI 447 - Machine Learning
//  Assignment #4, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  Module for the implementation of a Particle Swarm Optimization algorithm
//
//--------------------------------------------------------------------------------------------------------------

// MODULES
//--------------------------------------------------------------------------------------------------------------
    
// Open the modules
open Tools
open Tools.Extensions

// Declare as a module
module rec ParticleSwarmOptimization = 


// OBJECTS
//--------------------------------------------------------------------------------------------------------------
        
    // (none)
       

// FUNCTIONS
//--------------------------------------------------------------------------------------------------------------
       
    // Function to run Particle Swarm Optimization given a population
    let particleSwarmOpt (net : Network) =
        
        // Initialize the population as the layer ConnectionMatrix of the network
        let population = net.layers

        population
        |> Array.toSeq                                  // Convert the array to a sequence
        |> Seq.map (fun p ->                            // Set the best values
            {
                
                // Set the personal best
            
                // Set the regional (local) best
        
            })
        |> Seq.map (fun p ->                            // Update velocity and position
            {

                // Generate random r1 and r2

            
                // Set the velocity
                // v[i][j][t + 1] = v[i][j][t] + (c1 * r1 * (pb[i][j][t] - x[i][j][t])) + (c2 * r2 * (lb[i][j][t] - x[i][j][t]))


                // Set the position
                // x[i][j][t+1] = x[i][j][t] + v[i][j][t+1];
        
            })

// IMPLEMENTATIONS
//--------------------------------------------------------------------------------------------------------------
           
    let var1 = particleSwarmOpt
    
//--------------------------------------------------------------------------------------------------------------
// END OF CODE