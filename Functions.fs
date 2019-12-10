//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
//
//  CSCI 447 - Machine Learning
//  Assignment #4, Fall 2019
//  Chris Major, Farshina Nazrul-Shimim, Tysen Radovich, Allen Simpson
//
//  Functions for the implementation of several population algorithms to train a feedforward neural network
//
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 

// Assign to Project4 Namespace
namespace Project4

// Open the Types.fs file
open Types

// Create the functions module
module Functions =
    

    // TYPES
    //---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
    
    type Gene = float32                     // Type alias genes represet float32
    type Chromosome = Gene[,]               // Type alias Chromosomes represet 2D arrays of float32
    type Genome = Chromosome []             // Type alias Genomes represet Arrays of Chromosomes
    
    type Agent = {                          // Type for an array of individuals
        name                                    : string
        position                                : Genome                            // Current position
        pBest                                   : Genome                            // Personal best
        velocity                                : Genome                            // Current velocity
        neighbors                               : Agent[]                           // Population of neighbors
        bestError                               : float32                           // Best error seen by Agent
    }
    
    type Population = Agent []             // Type alias Genomes represet Arrays of individuals (Genomes)

    // RunGenerationOptions serves as metadata for a population
    type RunGenerationOptions =
        {
            sortByError                 : bool
            lossFn                      : float32[] -> float32[] -> float32
            nextGenFn                   : (Agent*float32)[] -> Agent[] option 
            converganceDeltaError       : float32
            maxGens                     : int 
        }


    // RANDOM NUMBER GENERATION FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to create a random number generator (RNG)
    let rand = System.Random()                           
    
    // Function to generate a random float32 within a normal distribution, given a mean and stdDev
    let randNorm mean stdDev = 
        let u1 = 1.0 - rand.NextDouble()                                                                    // Initialize random valuea and subtract from 1
        let u2 = 1.0 - rand.NextDouble()                                                                    // Initialize random valuea and subtract from 1
        let randStdNorm = sqrt(-2.0 * log(u1)) * sin(2.0 * System.Math.PI * u2);                            // We are using Box Muller Algorithm to generate random numbers
        (float mean) + (float stdDev) * randStdNorm |> float32                                              // Return as float

    let randUniform_m1_1() = (rand.NextDouble() |> float32)*(if rand.Next(2) = 0 then -1.f else 1.f)

    // Function to return the standard deviation of an array
    let getStdDev (s:float32[]) = 
        let mean = s|>Array.average                                                                         // Calculate the average of the array values
        let variance = s|>Array.sumBy(fun x -> (x - mean) * (x - mean))                                     // Calculate the variance                          
        let stdDev = sqrt variance                                                                          // Calculate the standard deviation    
        stdDev                                                                                              // Return the standard deviation
        
    // Function to return a random value as a float using NextDouble()
    let inline nextFloat32 () = rand.NextDouble() |> float32 


    // CHROMOSOME FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to get the height and width of a chromosome a
    let inline getHeightWidth (a:Chromosome) = a.GetLength(0), a.GetLength(1)

    // Function to verify if chromosomes a and b are the same size
    let validateChromosomeSize (a:Chromosome) (b:Chromosome) =
        if a.GetLength(0) <> b.GetLength(0) || a.GetLength(1) <> b.GetLength(1) then                        // If their lengths are not the same ...
            failwithf "a and b do not match!"                                                               // ... return failure message
        else 
            a.GetLength(0), a.GetLength(1)                                                                  // Otherwise, return lengths
    

    // CROSSOVER FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to perform uniform crossover of chromosomes a and b
    let uniformCrossover (a:Chromosome) (b:Chromosome) =
        let height, width = validateChromosomeSize a b                                                      // Take the height and width of the chromosomes
        let r1 = Array2D.zeroCreate height width                                                            // Create empty array of same size - child r1
        let r2 = Array2D.zeroCreate height width                                                            // Create empty array of same size - child r2
        for j = 0 to height - 1 do                                                                          // Iterate through array height    
            for i = 0 to width - 1 do                                                                       // Iterate through array width
                if rand.Next(2) = 0 then                                                                    // If the next random value is 0, swap a into r1 and b into r2
                    r1.[j,i] <- a.[j,i]
                    r2.[j,i] <- b.[j,i]
                else                                                                                        // Else, swap a into r2 and b into r1
                    r1.[j,i] <- b.[j,i]
                    r2.[j,i] <- a.[j,i]
        r1, r2                                                                                              // Return r1 and r2

    // Function to perform uniform crossover between genomes a and b
    let uniformCrossoverGenome (a:Genome) (b:Genome) =
        a                                                                                                   // Grab genome a
        |> Seq.zip b                                                                                        // Zip a and b into tuples
        |> Seq.map (fun (a,b) -> uniformCrossover a b)                                                      // Map a and b and perform uniform crossover
        |> Seq.toArray                                                                                      // Convert to array
        |> Array.unzip                                                                                      // Unzip into two new genomes

    // Function to perform uniform crossover of chromosomes a and b
    let pointCrossover (a:Chromosome) (b:Chromosome) =
        let height, width = validateChromosomeSize a b                                                      // Take the height and width of the chromosomes
        let pt = rand.Next(width*height)
        let jl = pt/width;
        let il = pt%width;
        let r1 = Array2D.zeroCreate height width                                                            // Create empty array of same size - child r1
        let r2 = Array2D.zeroCreate height width                                                            // Create empty array of same size - child r2
        for j = 0 to jl-1 do
            for i = 0 to width - 1 do
                r1.[j,i] <- a.[j,i]
                r2.[j,i] <- b.[j,i]
        for i = 0 to il-1 do
                r1.[jl,i] <- a.[jl,i]
                r2.[jl,i] <- b.[jl,i]
        for i = il to width-1 do                
                r1.[jl,i] <- b.[jl,i]
                r2.[jl,i] <- a.[jl,i]
        for j = jl to height - 1 do                                                                          // Iterate through array height    
            for i = 0 to width - 1 do                                                                       // Iterate through array width
                    r1.[j,i] <- b.[j,i]
                    r2.[j,i] <- a.[j,i]
        r1, r2                                                                                              // Return r1 and r2

    // Function to perform uniform crossover between genomes a and b
    let pointCrossoverGenome (a:Genome) (b:Genome) =
        a                                                                                                   // Grab genome a
        |> Seq.zip b                                                                                        // Zip a and b into tuples
        |> Seq.map (fun (a,b) -> pointCrossover a b)                                                      // Map a and b and perform uniform crossover
        |> Seq.toArray                                                                                      // Convert to array
        |> Array.unzip                                                                                      // Unzip into two new genomes


    // MUTATION FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to mutate a chromosome
    let mutation mutationRate (mean:float32[,]) (stdDev:float32[,]) chromosome =
        let height, width = getHeightWidth chromosome                                                                       // Grab the height and width of a chromosome
        for j = 0 to height - 1 do                                                                                          // Iterate through array height                                                                           
            for i = 0 to width - 1 do                                                                                       // Iterate through array width
                if rand.NextDouble() <= mutationRate then                                                                   // If our random value falls below the mutation rate ...
                    chromosome.[j,i]<- chromosome.[j,i] + ((*nextFloat32() *) (randNorm mean.[j,i] stdDev.[j,i])-mean.[j,i])              // ... mutate with a random value
                    //chromosome.[j,i]<- chromosome.[j,i] + randUniform_m1_1()                                                 // ... mutate with a random value

    // Function to mutate a genome
    let genomeMutation mutationRate (mean:float32[,][]) (stdDev:float32[,][]) genome =
        genome                                                                                                              // Take a genome                
        |> Seq.iteri (fun i c -> mutation mutationRate mean.[i] stdDev.[i] c)                                               // Iterate through each chromosome and mutate


    // NEURAL NETWORK FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to compute the dot product of matrix A and vector r
    // effectively r.[j] = SUM (i = 0 to height; x.[i]*A.[j,i]
    let dotProduct (x:float32[]) (A:float32[,]) (r:float32[]) =
        let height,width = getHeightWidth A                                                             // Get the height and width of A
        if width > x.Length || height > r.Length then                                                   // Confirm that our x array and r arrays are compatible with A
            failwithf "matrix is not the right size for x and r!"                                       // If not, kindly let us know
        
        for j = 0 to height-1 do                                                                        // Iterate over the rows (y axis)
            let mutable sum = 0.f                                                                       // Initiate a mutable sum
            for i = 0 to width-1 do                                                                     // Iterate over the columns (x axis)
                sum <- x.[i] * A.[j,i] + sum                                                            // Increment sum by x.[i]*A.[j,i]
            r.[j] <- sum                                                                                // Write into then jth element of r the sum
        
    // Logistic Function
    let logistic (x:float32 []) (r:float32[]) = 
        let inline logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) ))|>float32                 // We are using the logistic function 1/(1+e^-x)
        for i = 0 to x.Length-1 do                                                                      // Iterate through x 
            r.[i]<- logistic x.[i]                                                                      // And write into the ith element of r the logistic function applied to ith element of x 

    // Function to run the feedforward network
    let runFeedForward (genome:Genome) (inputs:float32[]) =
        let _,width = getHeightWidth genome.[0]                                                         // Get the width of the genome's first chromosome
        if width <> inputs.Length then                                                                  // Confirm that the width is compatable with the input array
            failwithf "matrix is not the right size for x and r!"                                       // If not let me know...
        let maxLen = genome|>Seq.map(fun c -> max (c.GetLength(0)) (c.GetLength(1)))|>Seq.max           // Get the maximum length of any layer within the genome
        let b1,b2 = Array.zeroCreate maxLen , Array.zeroCreate maxLen                                   // Create 2 buffer arrays of maxlen size to store the results of feeding forward
        genome                                                                                          // Now the MAGIC:
        |> Seq.fold (fun (inputBuffer,useb1) c ->                                                       // Fold over the elements of genome (chromosomes) and track the state (a tuple of in input buffer and a boolean telling us to use b1 or b2)
            let output = if useb1 then b1 else b2                                                       // Set our output to b1 if we have a true bool, or b2 if a false one.
            dotProduct inputBuffer c output                                                             // Then we apply the dot product function (see above) on inputbuffer and our chromosome, writing into output
            logistic output output                                                                      // Then we apply the logistic function (also see above) on our output writing back into output
            output,(not useb1)                                                                          // Then push the output ahead and the negation of the last "useb1" value.
        ) (inputs,true)                                                                                 // Our initial stlate will be using the inputs array and true so we write into b1 as our first output
        |>fst                                                                                           // Then we strip off the booleans giving us just the output layer


    // LOSS FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     
       
    //CountingLoss
    let countingLoss (predicted:float32[]) (expected:float32[]) =
       Seq.zip predicted expected                                                                       // Zip the arrays of predicted and expected values into tuples
       |> Seq.map (fun (predicted,expected) ->                                                          // Map through each tuple            
           let predicted = if predicted < 0.5f then false else true                                     // Represent the predicted value as a float
           let expected =  if expected  = 0.f then false else true                                      // Represent the expected value as a bool
           if predicted = expected then 0.0f else 1.0f
           //abs(predicted-expected)

       ) |> Seq.sum                                                                                     // Sum each loss and return

    // Mean Square Error Loss
    let MSELoss (predicted:float32[]) (expected:float32[])=
        let length = min expected.Length predicted.Length
        let mutable errSum = 0.f                                                                        // Initialize mutable sum value
        for i = 0 to length-1 do                                                              // Iterate through predicted values
            errSum <- let d = predicted.[i] - expected.[i] in d*d+errSum                                // Square the difference between the output and expected node and sum it
        errSum/(float32 predicted.Length)                                                               // Divide by the number of predicted elements to make the derivitave easier

    // Function to calculate the error based on a given loss function
    let calculateError lossFunction predicted expected : float32 =
        let loss = lossFunction predicted expected                                                                 // Return the loss of the predicted/expected pair
        if System.Single.IsNaN loss then 
            failwithf "error is not a number!"
        else loss


    // POPULATION ALGORITHM FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Functions to initialize chromosomes - constant or random
    let initChromosome h w = Array2D.init h w (fun _ _ -> 1.f)
    let initRandChromosome rand h w = Array2D.init h w (fun _ _ -> rand())

    // Functions to initialize genomes
    let initGenome sizes = 
        sizes |> Seq.pairwise |> Seq.map (fun (i,o) -> initChromosome o i) |> Seq.toArray               // Initialize a genome with constant chromosomes
    
    let initRandGenome rand sizes = 
        sizes |> Seq.pairwise |> Seq.map (fun (i,o) -> initRandChromosome rand o i) |> Seq.toArray      // Initialize a genome with random chromosomes

    // Function to get the mean and standard deviation of a genome
    let getGenomeMeanAndStdDev (pop:seq<Genome>) =
        let mutable cnt = 0                                                                             // Create a mutable count value
        let firstEntry = pop |> Seq.head
        let mean =                                                                                      // Calculate the mean by ...
            let sum =                                                                                   // Create summation buffer that is the same size as the given genomes
                Array.init firstEntry.Length (fun i ->                                          
                    let h,w = firstEntry.[i].GetLength(0), firstEntry.[i].GetLength(1)                  // Get the size of the first entry
                    Array2D.zeroCreate h w)                                                             // Create an array filled with 0.f of same size
            pop                                                         
            |> Seq.iter (fun genome ->                                                                  // Compute the accumulation of the genomes in the sequence
                cnt <- cnt + 1                                                                          // Increment the count value        
                genome
                |> Seq.iteri (fun i chromosome ->                                                       // Iterate through the chromosomes of a genome    
                    let h,w = chromosome.GetLength(0), chromosome.GetLength(1)                          // Get the height and width
                    let sum_i = sum.[i]                                                                 // Grab the sum at index i
                    for j = 0 to h-1 do                                                                 // Iterate through array height          
                        for i = 0 to w-1 do                                                             // Iterate through array width
                            sum_i.[j,i] <- chromosome.[j,i] + sum_i.[j,i]                               // Add i_jth value of chromosome to the i_jth value of sum and write backinto the i_jth value of sum
                )
            )
            let cnt = float32 cnt                                                                       // Convert the integer cnt into a float32
            sum                                                                                         // change the values in sum into thier means by dividing the entries by cnt
            |> Seq.iter (fun sum ->                                                                     // Iterate through sum values
                let h,w = sum.GetLength(0), sum.GetLength(1)                                            // get then height and width of the sum 
                for j = 0 to h-1 do                                                                     // Iterate through array height          
                    for i = 0 to w-1 do                                                                 // Iterate through array width
                        sum.[j,i] <- sum.[j,i]/cnt                                                      // override the j_ith value of sum with the j_ith value/ the count (calculated earlier)
            )
            sum                                                                                         //return sum (which is actually an array of 2D Arrays of means... (or a Genome of means (the most average dude)))

        let stdDev =                                                                                    // Calculate the standard deviation by ...
            let cnt = float32 cnt                                                                       // Create a count value
            let buf : float32[,][] =                                                                    // Create a 2D array of float values
                Array.init mean.Length (fun i ->                                                        // Initialize an array of values
                                            let h,w = mean.[i].GetLength(0), mean.[i].GetLength(1)      // Grab the means
                                            Array2D.zeroCreate h w)                                     // Create an array off of it

            pop                                                                                         // Compute variance            
            |> Seq.iter (fun genome ->
                genome                                                                                  // Grab the genome
                |> Seq.iteri (fun i chromosome ->                                                       // Iterate through each chromosome    
                    let h,w = chromosome.GetLength(0), chromosome.GetLength(1)                          // Grab the height and width
                    let buf_i = buf.[i]                                                                 // Get the index value of the 2D array
                    let mean_i = mean.[i]                                                               // Grab the average at that index
                    for j = 0 to h-1 do                                                                 // Iterate through array height  
                        for i = 0 to w-1 do                                                             // Iterate through array width
                            let x = chromosome.[j,i]                                                    // Grab the chromosome
                            let mean = mean_i.[j,i]                                                     // Grab the mean
                            let x_mean = x-mean                                                         // Recalculate the mean
                            buf_i.[j,i] <- x_mean*x_mean + buf_i.[j,i]                                  // Add into the 2D array
                )
            )
            
            buf                                                                                         // Compute stddev
            |> Seq.iter (fun buf ->
                let h,w = buf.GetLength(0), buf.GetLength(1)
                for j = 0 to h-1 do                                                                     // Iterate through array height  
                    for i = 0 to w-1 do                                                                 // Iterate through array width
                        buf.[j,i] <- sqrt (buf.[j,i]/cnt)                                               // Calculate the standard deviation
            )
            buf                                                                                         // Return the 2D array

        mean, stdDev                                                                                    // Return the mean and standard deviation 


    // PARTICLE SWARM OPTIMIZATION FUNCTIONS
    //--------------------------------------------------------------------------------------------------------------

    // Function to update the position of a member
    let updateAgentPosition (agent : Agent) =
        
        for i = 0 to agent.position.Length-1 do                                           // Iterate through the agent's position genome length
            for j = 0 to agent.position.[i].GetLength(0)-1 do                             // Iterate through the agent's position genome's chromosome length
                for k = 0 to agent.position.[i].GetLength(1)-1 do                         // Iterate through the agent's position genome's chromosome width
                    let pos = agent.position.[i].[j,k]                                  // Get the agent's position genome's chromosome
                    let vel = agent.velocity.[i].[j,k]                                  // Get the agent's velocity genome's chromosome
                    agent.position.[i].[j,k] <- pos + vel                               // Add

        
        // agent.position = agent.position + agent.velocity                            // Update position

    // Function to update the position of a member
    let updateAgentVelocity agent (omega : float32) (c1 : float32) (c2 : float32) (gBest : Genome) =

        let r1 = nextFloat32()                                                          // Generate random value r1
        let r2 = nextFloat32()                                                          // Generate random value r2

        for i = 0 to agent.position.Length - 1 do                                           // Iterate through the agent's position genome length
            for j = 0 to agent.position.[i].GetLength(0) - 1 do                             // Iterate through the agent's position genome's chromosome length
                for k = 0 to agent.position.[i].GetLength(1) - 1 do                         // Iterate through the agent's position genome's chromosome width
                    let vel = agent.velocity.[i].[j,k]                                  // Get the agent's velocity genome's chromosome
                    let pos = agent.position.[i].[j,k]
                    let pBest = agent.pBest.[i].[j,k]
                    let g = gBest.[i].[j,k]

                    agent.velocity.[i].[j,k] <- (omega * vel) + (c1 * r1 * (pBest - pos)) + (c2 * r2 * (g - pos))       // Recalculate




    // RUNTIME FUNCTIONS
    //----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------     

    // Function to run the training set of the data
    let runTrainingSet lossFunction (g:Agent) (trainingSet:float32[][]) (expectedResults:float32[][]) =
        trainingSet                                                                                             // Grab the training set                        
        |>Seq.zip expectedResults                                                                               // Zip with the expected results into tuples
        |>Seq.sumBy (fun (er,ts) ->                                                                             // Average the tiple values
            let pred = runFeedForward g.position ts                                                             // Run the feedforward neural network        
            calculateError lossFunction pred er                                                                 // Calculate the error of the results    
        )

    // Function to run a generation
    let runGeneration (options:RunGenerationOptions) (initialPopulation:Population) (trainingSet:float32[][]) (expectedResults:float32[][])  =
        let rec loop cnt (lastMinErr:float32) (currentPopulation:Population) =                                  // Loop through the current population
            let popsWithErrors =                                                                                // Grab the population members with errors
                let sw = System.Diagnostics.Stopwatch.StartNew()
                // currentPopulation                                                                               // For the current population ...
                // |>Array.map (fun a -> a,runTrainingSet options.lossFn a trainingSet expectedResults)            // ... map out the training set
                // |>(if  options.sortByError then Array.sortBy snd else id)                                       // Sort based on error
                let results =
                    currentPopulation
                    |> Seq.map (fun a -> async { return a,runTrainingSet options.lossFn a trainingSet expectedResults })
                    |> Async.Parallel
                    |> Async.RunSynchronously
                    |>(if  options.sortByError then Array.sortBy snd else id)                                       // Sort based on error
                sw.Stop()
                let seconds = sw.Elapsed.TotalSeconds
                printfn "\nGen[%02d]: elapsedTime: %f time per training set: %f" cnt seconds (seconds/float trainingSet.Length)
                results
            printfn "best: %s err: %f" (popsWithErrors.[0]|>fst).name (popsWithErrors.[0]|>snd)
            let minErr = popsWithErrors |> Seq.map snd |> Seq.min                                               // Grab the smallest error
            let errDelta = abs(lastMinErr-minErr)                                                               // Calculate the delta value
            let avgErr = popsWithErrors |> Array.averageBy snd                                                  // Calculate the average error
            let bestErr = popsWithErrors |> Seq.map snd |> Seq.min                                              // Calculate the best error
            let worstErr = popsWithErrors |> Seq.map snd |> Seq.max                                             // Calculate the worst error            
            if (cnt+1)>options.maxGens then
                printfn "Max Generations Reached: %d\naverage error: %f best error: %f worst error: %f" options.maxGens avgErr bestErr worstErr
                popsWithErrors |> Seq.minBy snd |> fst
            else
                if errDelta > options.converganceDeltaError then                                                    // If the error delta is greater than the convergence delta ...
                    match options.nextGenFn popsWithErrors with                                                     // Print error messages
                    | None ->
                        printfn "Gen[%02d] Next Gen Function Stopped.\naverage error: %f best error: %f worst error: %f" cnt avgErr bestErr worstErr
                        popsWithErrors |> Seq.minBy snd |> fst
                    | Some v ->
                        printfn "Gen[%02d] errDelta: %f average error: %f best error: %f worst error: %f" cnt errDelta avgErr bestErr worstErr
                        loop (cnt+1) minErr v
                else
                    printfn "Gen[%02d] Local Optimal Reached: %f < %f\naverage error: %f best error: %f worst error: %f" cnt errDelta options.converganceDeltaError avgErr bestErr worstErr
                    popsWithErrors |> Seq.minBy snd |> fst 
        loop 0 System.Single.MaxValue initialPopulation

    // Function to run a Genetic Algorithm (GA) on the next generation
    let mutable nextChild = 0
    let simpleGANextGen mutationRate (popErrs:(Agent*float32)[]) =
        let maxidx = popErrs.Length/3                                                                           // Grab the max ID index
        let mean,stdDev = popErrs |> Seq.map (fun x -> (fst x).position)|> getGenomeMeanAndStdDev               // Get the mean and standard deviation
        let p = Array.zeroCreate popErrs.Length                                                                 // Create an array to represent the current population
        let mutable p_i = 0                                                                                     // Create an array to represent the next population
        while p_i < (maxidx*2) do                                                                                             
            let a,_ = popErrs.[rand.Next(maxidx+1)]                                                             // Randomly grab members
            let b,_ = popErrs.[rand.Next(maxidx+1)]                                                             // Randomly grab members
            
            let c_1,c_2 =                                                                                       // Create two children
                //uniformCrossoverGenome a.position b.position                                                    // Perform crossover
                pointCrossoverGenome a.position b.position
            genomeMutation mutationRate mean stdDev c_1                                                         // Mutate child 1
            genomeMutation mutationRate mean stdDev c_2                                                         // Mutate child 2
            let ac_1 = {a with position = c_1; name =  string nextChild} 
            nextChild <- nextChild+1
            let ac_2 = {b with position = c_2; name =  string nextChild} 
            nextChild <- nextChild+1
                

            p.[p_i] <- ac_1                                                                                     // Update the population ...
            p_i <- p_i+1
            if p_i < p.Length then
                p.[p_i] <- ac_2                                                                                 // ... with the child
                p_i <- p_i+1                                                                                    // ... with the previous member
        let mutable ii = 0            
        while p_i < p.Length do
            p.[p_i] <- let x,_ = popErrs.[ii] in x
            p_i <- p_i + 1
            ii <- ii + 1
        Some p  
       
       
    // Function for performing PSO
    let simplePSO (popErrs:(Agent*float32)[]) = 
        let omega = 1.0f                                                                                        // Assign omega
        let c1 = 1.0f                                                                                           // [PSO] Assign c1
        let c2 = 1.0f                                                                                           // [PSO] Assign c2
        let gBest = popErrs |> Seq.map fst |> Seq.head |> fun x -> x.pBest                                      // Get gBest from all possible pBests                                         // Update gBest            
        popErrs                                                                                                 // Perform PSO
        |> Seq.map (fun (a, e) ->                                                                               // Map through the population's errors
            let a = {a with pBest =if a.bestError > e then a.position else a.pBest }
            updateAgentPosition a                                                                               // [PSO] Update position
            updateAgentVelocity a omega c1 c2 gBest                                                             // [PSO] Update velocity
            a                                                                                                   // Return a
        ) |> Seq.toArray |> (fun x -> match x.Length with    
            | 0 -> None
            | _ -> Some x
        )
            

    // Function to initialize a population
    let initializePopulatation (genomeDescriptor:seq<int>) size =
        Array.init size (fun i ->                                                                               // Initialize an array ...
            let pos = initRandGenome randUniform_m1_1 genomeDescriptor                                                    // ... generate random genomes
            let name = string nextChild
            nextChild <- nextChild+1
            {                          // Type for an array of individuals
                name = name
                position = pos
                pBest = pos                                                    // ... generate random genomes
                velocity = initRandGenome (randUniform_m1_1 >> (fun x -> x*0.25f)) genomeDescriptor                                                    // ... generate random genomes
                neighbors = Array.zeroCreate 0
                bestError = 0.0f
            }
            
        )


//---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
// END OF CODE