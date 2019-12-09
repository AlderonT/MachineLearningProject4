namespace Project4 

open Types

module Util =

    let createRandNorm mean stdDev = 
        let rand = System.Random()
        fun () -> 
            let u1 = 1.0-rand.NextDouble()
            let u2 = 1.0-rand.NextDouble()
            let randStdNorm = sqrt(-2.0 * log(u1)) * sin(2.0 * System.Math.PI * u2); //random normal(0,1)
            (float mean) + (float stdDev) * randStdNorm |> float32

    let selectRand (s:'a[]) = 
        let idx = System.Random().Next(0,s.Length-1)
        s.[idx]

    let selectRandStdDev (s:float32[]) = 
        let mean = s|>Array.average 
        let variance = s|>Array.sumBy(fun x -> (x-mean)*(x-mean)) 
        let stdDev = sqrt variance
        createRandNorm mean stdDev ()

    let rnd = System.Random()
    
    ////CROSSOVER:
    // Pop -> Pop -> (Pop*Pop) -- we are doing uniform crossover
    let crossover (mother:Pop) (father:Pop) = 
        // the variable was typed here for intellisense reasons
        let (crossed:(float32*float32)[][]) = 
            mother.chromosomes 
            |> Array.zip father.chromosomes     // float32[][] -> float32[][] -> (float32[]*float32[])[] -- we are pairing up the chromosomes for mother and father. 
            |> Array.map (fun ((a,b) : float32[]*float32[]) -> Array.zip a b) // (float32[]*float32[])[] -> (float32*float32)[][] -- we are pairing up the genes for mother and father
            |> Array.map (fun c -> c |> Array.map (fun ((a,b):float32*float32) -> if rnd.Next(0,1) = 0 then a,b else b,a))  // (float32*float32)[][] -- we iterate through the chromosomes and in each chromosome we map each value to a function where, if the random number is 0 we leave the chromosome pair alone, and if it is one we cross them.
        //build the first child
        let popA = {
            chromosomes = (crossed |> Array.map (fun c -> c |> Array.map fst ))         //create a tuple of pops were we take crossed and grab only the first, and only the first weight
            neighbors = if rnd.Next(0,1) = 0 then mother.neighbors else father.neighbors//rather than regenerating these values, since they are not used for any algorithm using crossover, they are selected randomly between the two parents 
            velocity = if rnd.Next(0,1) = 0 then mother.velocity else father.velocity   //rather than regenerating these values, since they are not used for any algorithm using crossover, they are selected randomly between the two parents
            pBest = (crossed |> Array.map (fun c -> c |> Array.map fst ))               //the pBest is set to the child's chromosomes
            pBestFitness = System.Single.MaxValue                                       //pBest's fitness should be set to the max value since this new child has not generated it's fitness.
            metadata = mother.metadata                                                  //metadata SHOULD be the same for mother and father
        }
        //build the second child
        let popB = {
            chromosomes = (crossed |> Array.map (fun c -> c |> Array.map snd))          //create a tuple of pops were we take crossed and grab only the first, and only the second weight
            neighbors = if rnd.Next(0,1) = 0 then mother.neighbors else father.neighbors//rather than regenerating these values, since they are not used for any algorithm using crossover, they are selected randomly between the two parents 
            velocity = if rnd.Next(0,1) = 0 then mother.velocity else father.velocity   //rather than regenerating these values, since they are not used for any algorithm using crossover, they are selected randomly between the two parents
            pBest = (crossed |> Array.map (fun c -> c |> Array.map snd ))               //the pBest is set to the child's chromosomes
            pBestFitness = System.Single.MaxValue                                       //pBest's fitness should be set to the max value since this new child has not generated it's fitness.
            metadata = mother.metadata                                                  //metadata SHOULD be the same for mother and father
        }
        popA,popB   //return both children as a tuple.

    //// MUTATION:
    // Because weights are all real values we are using creep gaussian mutation for the 
    // Creep Gaussian: X_i' = X_i + N(0, sigma_i)
    // N(x,y) = rand(x,y) -- inclusive        

    let mutate (mutationRate:float) (individual:Pop) = 
        for i = 0 to individual.chromosomes.Length - 1 do                                          
            for j = 0 to individual.chromosomes.[i].Length - 1 do 
                if rnd.NextDouble() <= mutationRate  then  //if we get a number that is less then our mutation rate (or the mutation rate is 100% (NextDouble() only gives values 0.0 - 1.0 and we want a 100% mutation rate to always mutate) do the mutation
                    let sigma_i_j = 
                        individual.neighbors.pops |> Array.map (fun p -> p.chromosomes.[i].[j])                    // we go through each neighbor to pop and get their i_jth gene (in their chromosome) 
                        |> selectRandStdDev                                                                 // and select a number based on a standard deviation as defined above
                    let x_i_j' = individual.chromosomes.[i].[j] + (float32 (rnd.NextDouble()*(sigma_i_j|>float)))  // since rnd produces a number between 0.0 and 1.0 we can multiply the resuting number by sigma_i_j to get a number between 0.0 and sigma_i_j
                    individual.chromosomes.[i].[j] <- x_i_j'                                                       // then we re-assign the i_jth gene in pop to equal x_i_j'

    

    //Random rand = new Random(); //reuse this if you are generating many
    //double u1 = 1.0-rand.NextDouble(); //uniform(0,1] random doubles
    //double u2 = 1.0-rand.NextDouble();
    //double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
    //             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
    //double randNormal =
    //             mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)