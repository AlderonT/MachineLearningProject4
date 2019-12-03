namespace Project4 
module rec Types =

// OBJECTS
//--------------------------------------------------------------------------------------------------------------
        
    // Create a Metadata object to distinguish real and categorical attributes by index
    type DataSetMetadata = 
        abstract member getRealAttributeNodeIndex           : int -> int            // Indices of the real attributes
        abstract member getCategoricalAttributeNodeIndices  : int -> int[]          // Indices of the categorical attributes
        abstract member inputNodeCount                      : int                   // number of input nodes
        abstract member outputNodeCount                     : int                   // number of output nodes
        abstract member getClassByIndex                     : int -> string         // get the class associated with this node's index
        abstract member fillExpectedOutput                  : Point -> float32[] -> unit    //assigned the expected output of Point to the float32[]
        abstract member isClassification                    : bool                  //stores if this dataset is classification

    // Create a Layer object to represent a layer within the neural network
    type Layer = {
        nodes                                   : float32[]                         // Sequence to make up vectors
        nodeCount                               : int                               // Number of nodes in the layer
        deltas                                  : float32[]                         // sacrificing space for speed
    }
    // Create a ConnectionMatrix object to represent the connection matrix within the neural network
    type ConnectionMatrix = {
        weights                                 : float32[]                         // Sequence of weights within the matrix
        inputLayer                              : Layer                             // Input layer
        outputLayer                             : Layer                             // Output layer
    }
    // Create a Network object to represent a neural network
    type Network = {
        layers                                  : Layer[]                           // Array of layers within the network
        connections                             : ConnectionMatrix[]                // Array of connections within the network
    }
        with 
            member this.outLayer = this.layers.[this.layers.Length-1]
            member this.inLayer = this.layers.[0]

    // Create a Point object to represent a point within the data
    type Point = {
        realAttributes                          : float32[]                         // the floating point values for the real points
        categoricalAttributes                   : int[]                             // the values for categorical attributes. distance will be discrete
        cls                                     : string option
        regressionValue                         : float option
        metadata                                : DataSetMetadata
    }

        // Method for the Point object (not entirely functional but simple enough for application)
        with 
            member this.distance p = //sqrt((Real distance)^2+(categorical distance)^2) //categorical distance = 1 if different value, or 0 if same value
                (Seq.zip this.realAttributes p.realAttributes|> Seq.map (fun (a,b) -> a-b)|> Seq.sumBy (fun d -> d*d))
                + (Seq.zip this.categoricalAttributes p.categoricalAttributes |> Seq.sumBy (fun (a,b)-> if a=b then 0.f else 1.f))
                |>sqrt 
    

    type Pop = {
        chromosomes             : float32[][]
        neighbors               : Population
        velocity                : float32[][]
        pBest                   : float32[][]
        pBestFitness            : float32
        metadata                : DataSetMetadata
    }
        with 
            member x.calculateFitness trainingSet = 
                let network = createNetworkFromPop x.metadata x.chromosomes
                let MSE =
                    trainingSet
                    |> Seq.map ( fun p ->
                        runNetwork metadata network p 
                        |> fun (_,_,err) -> err
                    )
                    |>Seq.average
                -MSE
            member x.updateVelocity = 
                let rnd = System.Random()
                fun () ->
                    let omega = 0.2f //tune inertia!
                    let phi_1 = 0.5f * (rnd.NextDouble()|>float32) //tune the float
                    let phi_2 = 0.5f * (rnd.NextDouble()|>float32) //tune the float
                    x.velocity
                    |> Array.mapi (fun i c -> c |> Array.mapi (fun j (v:float32) -> (omega * v) + (phi_1*(x.pBest.[i].[j]-x.chromosomes.[i].[j])) + (phi_2 * (x.neighbors.gBest()-x.chromosomes.[i].[j])))) // v(t) = omega * v(t-1) + c_1 * r_1 * (pBest - x(t)) + c_2 * r_2 * (gBest - x(t))  

    type Population = {
        pops                    : Pop[]
    }
        with 
            member p.gBest =  p.pops |> Array.maxBy (fun x -> x.pBestFitness)

    
