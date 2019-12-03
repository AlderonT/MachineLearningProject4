namespace Project4
open Types
module Functions =
    
    type Gene = float32
    type Chromosome = Gene[,]
    type Genome = Chromosome []
    
    let rand = System.Random()

    let randNorm mean stdDev = 
        let u1 = 1.0-rand.NextDouble()
        let u2 = 1.0-rand.NextDouble()
        let randStdNorm = sqrt(-2.0 * log(u1)) * sin(2.0 * System.Math.PI * u2); //random normal(0,1)
        (float mean) + (float stdDev) * randStdNorm |> float32

    let getStdDev (s:float32[]) = 
        let mean = s|>Array.average 
        let variance = s|>Array.sumBy(fun x -> (x-mean)*(x-mean)) 
        let stdDev = sqrt variance
        stdDev

    let inline nextFloat32 () = rand.NextDouble() |> float32 

    let inline getHeightWidth (a:Chromosome) = a.GetLength(0),a.GetLength(1)

    let validateChromosomeSize (a:Chromosome) (b:Chromosome) =
        if a.GetLength(0) <> b.GetLength(0) || a.GetLength(1) <> b.GetLength(1) then 
            failwithf "a and b do not match!"
        else 
            a.GetLength(0),a.GetLength(1)


    let uniformCrossover (a:Chromosome) (b:Chromosome) =
        let height,width = validateChromosomeSize a b 
        let r1 = Array2D.zeroCreate height width
        let r2 = Array2D.zeroCreate height width
        for j = 0 to height-1 do  
            for i = 0 to width-1 do
                if rand.Next(2) = 0 then 
                    r1.[j,i] <- a.[j,i]
                    r2.[j,i] <- b.[j,i]
                else
                    r1.[j,i] <- b.[j,i]
                    r2.[j,i] <- a.[j,i]
        r1,r2

    let mutation mutationRate (mean:float32[,]) (stdDiv:float32[,]) chromosome =
        let height,width = getHeightWidth chromosome
        for j = 0 to height-1 do  
            for i = 0 to width-1 do
                if rand.NextDouble()<=mutationRate then 
                    chromosome.[j,i]<- chromosome.[j,i] + (nextFloat32() * (randNorm mean.[j,i] stdDiv.[j,i])) 
    
    let dotProduct (x:float32[]) (A:float32[,]) (r:float32[]) =
        let height,width = getHeightWidth A 
        if width > x.Length || height > r.Length then  
            failwithf "matrix is not the right size for x and r!"
        
        for j = 0 to height-1 do  
            let mutable sum = 0.f
            for i = 0 to width-1 do
                sum <- x.[i] * A.[j,i] + sum
            r.[j] <- sum
        
    let logistic (x:float32 []) (r:float32[]) = 
        let inline logistic (x:float32) = (1./(1.+System.Math.Exp(float -x) ))|>float32    //Logistic Fn
        for i = 0 to x.Length-1 do 
            r.[i]<- logistic x.[i]


    let runFeedForward (genome:Genome) (inputs:float32[]) =
        let height,width = getHeightWidth genome.[0] 
        if width <> inputs.Length then  
            failwithf "matrix is not the right size for x and r!"
        let maxLen = genome|>Seq.map(fun c -> max (c.GetLength(0)) (c.GetLength(1)))|>Seq.max
        let b1,b2 = Array.zeroCreate maxLen , Array.zeroCreate maxLen
        genome
        |> Seq.fold (fun (inputBuffer,useb1) c -> 
            let output = if useb1 then b1 else b2 
            dotProduct inputBuffer c output
            logistic output output
            output,(not useb1)
        ) (inputs,true)
        |>fst

    let crossEntropyLoss (predicted:float32[]) (expected:float32[]) =
       Seq.zip predicted expected
       |> Seq.map (fun (predicted,expected) ->
           let predicted = float predicted
           let expected = float expected
           let log = System.Math.Log2
           -(expected*(log predicted) + (1.- expected)*(log (1. - predicted)))
           |> float32
       ) |> Seq.sum
 
    let MSELoss (predicted:float32[]) (expected:float32[])=
        let mutable errSum = 0.f
        for i = 0 to predicted.Length-1 do 
            errSum <- let d = predicted.[i] - expected.[i] in d*d+errSum                         //we square the difference between the output and expected node and sum it
        errSum/(float32 predicted.Length)                                                        //then divide by 2 to make the derivitave easier

    let calculateError lossFunction predicted expected : float32 =
        lossFunction predicted expected

    type Population = Genome []

    //let getGeneStdDivAndMean (pop:seq<Genome>) =
    //    getStdDev

    let runTrainingSet lossFunction (g:Genome) (trainingSet:float32[][]) (expectedResults:float32[][])=
        trainingSet
        |>Seq.zip expectedResults
        |>Seq.averageBy (fun (er,ts) -> 
            let pred = runFeedForward g ts  
            calculateError lossFunction pred er
        )

    type RunGenerationOptions =
        {
            sortByError         : bool
            lossFn              : float32[] -> float32[] -> float32
            nextGenFn           : (Genome*float32)[] -> Genome[] option 
            converganceDeltaError    : float32
        }

    let runGeneration (options:RunGenerationOptions) (initialPopulation:Population) (trainingSet:float32[][]) (expectedResults:float32[][])  =
        let rec loop (lastMinErr:float32) currentPopulation =
            let popsWithErrors =
                currentPopulation
                |>Array.map (fun g -> g,runTrainingSet options.lossFn g trainingSet expectedResults)
                |>(if  options.sortByError then Array.sortBy snd else id)
            let minErr = popsWithErrors |> Seq.map snd |> Seq.min
            if abs(lastMinErr-minErr) > options.converganceDeltaError then 
                match options.nextGenFn popsWithErrors with 
                | None -> popsWithErrors |> Seq.minBy snd |> fst
                | Some v -> printfn "average error: %f" (popsWithErrors |>Array.averageBy snd); loop minErr v
            else popsWithErrors |> Seq.minBy snd |> fst 
        loop System.Single.MaxValue initialPopulation

    let simpleGANextGen mutationRate (popErrs:(Genome*float32)[]) =
        let maxidx = popErrs.Length/2
        let p = Array.zeroCreate popErrs.Length
        let mutable p_i = 0
        while p_i < p.Length do
            let a,_ = popErrs.[rand.Next(maxidx+1)]
            let b,_ = popErrs.[rand.Next(maxidx+1)]
            
            let c_1,c_2 = 
                a 
                |> Array.zip b 
                |> Array.map (fun (a,b) -> uniformCrossover a b)
                |> Array.unzip 
            
            







    let calculateFitness trainingSet  (x:Pop) = 
        let network = createNetworkFromPop x.metadata x.chromosomes
        let MSE =
            trainingSet
            |> Seq.map ( fun p ->
                runNetwork metadata network p 
                |> fun (_,_,err) -> err
            )
            |>Seq.average
        -MSE
    let calculateVelocity = 
        let rnd = System.Random()
        fun (x:Pop) ->
            let omega = 0.2f //tune inertia!
            let phi_1 = 0.5f * (rnd.NextDouble()|>float32) //tune the float
            let phi_2 = 0.5f * (rnd.NextDouble()|>float32) //tune the float
            x.velocity
            |> Array.mapi (fun i c -> c |> Array.mapi (fun j (v:float32) -> (omega * v) + (phi_1*(x.pBest.[i].[j]-x.chromosomes.[i].[j])) + (phi_2 * (x.neighbors.gBest()-x.chromosomes.[i].[j])))) // v(t) = omega * v(t-1) + c_1 * r_1 * (pBest - x(t)) + c_2 * r_2 * (gBest - x(t))  
