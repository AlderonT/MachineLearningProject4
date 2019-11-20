module Util

    let createRandNorm mean stdDev = 
        let rand = System.Random()
        fun () -> 
            let u1 = 1.0-rand.NextDouble()
            let u2 = 1.0-rand.NextDouble()
            let randStdNorm = sqrt(-2.0 * log(u1)) * sin(2.0 * System.Math.PI * u2); //random normal(0,1)
            mean + stdDev * randStdNorm

    let rand = createRandNorm 0.0 1.0
    rand()

    let selectRand (s:'a[]) = 
        let idx = System.Random().Next(0,s.Length-1)
        s.[idx]
    let selectRandStdDev (s:float[]) = 
        let mean = s|>Array.average 
        let variance = s|>Array.sumBy(fun x -> (x-mean)*(x-mean)) 
        let stdDev = sqrt variance
        createRandNorm mean stdDev ()
    let rnd = System.Random()
    let s = 
        Seq.init 1000 (fun _ -> rnd.NextDouble())
        |>Seq.toArray
    selectRandStdDev s


    //Random rand = new Random(); //reuse this if you are generating many
    //double u1 = 1.0-rand.NextDouble(); //uniform(0,1] random doubles
    //double u2 = 1.0-rand.NextDouble();
    //double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
    //             Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
    //double randNormal =
    //             mean + stdDev * randStdNormal; //random normal(mean,stdDev^2)