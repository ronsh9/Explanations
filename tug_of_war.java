var strengthSD = 20

var strengthDistribution = Gaussian({mu: 50, sigma: strengthSD})

var unwell = mem(function(person){return flip(0.1)})

var strength = mem(function(person) {
  var a = sample(strengthDistribution)
  return unwell(person)? 0.5*a:a
})

var totalPulling = function(team) {
  return sum(map(function(person) {
    return strength(person)
  }, team))
}

var winner = function(team1, team2) {
  return totalPulling(team1)>totalPulling(team2)? 1 : 2
}

////////////////////////////////////////////////////////////////////

var MyWorldModel = function() {
    condition(strength("Anna")<30)  
    condition(unwell("Bob")==true)
  return {my_prediction_for_George_winning: winner(["Anna","Bob"],["George"])==2}
}
var KerenWorldModel = function() {  
    condition(unwell("Bob")==true)
  return {Keren_prediction_for_George_winning: winner(["Anna","Bob"],["George"])==2}
}
var JaneWorldModel = function() {  
    condition(strength("Anna")<30)
  return {Keren_prediction_for_George_winning: winner(["Anna","Bob"],["George"])==2}
}
var BobKerenWorldModel = function() {  
    condition(unwell("Bob")==true)
  return {George_wins: winner(["Anna","Bob"],["George"])==2}
}
var AnnaKerenWorldModel = function() {  
    condition(unwell("Bob")==true)
    condition(strength("Anna")<30)
  return {George_wins: winner(["Anna","Bob"],["George"])==2}
}

////////////////////////////////////////////////////////// anna is weak

var distKerenWorldModel = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   KerenWorldModel)
viz(distKerenWorldModel)

var distAnnaKerenWorldModel = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   AnnaKerenWorldModel)
viz(distAnnaKerenWorldModel)


// old:
//explanation score for answering "Anna is weak" for Keren
// var expl_score = Math.exp(distAnnaKerenWorldModel.score(false))- Math.exp(distKerenWorldModel.score(false))
// console.log(expl_score)

// Probability of George winning when considering both Bob's illness and Anna's weakness
var probGeorgeWinsAnnaWeak = Math.exp(distAnnaKerenWorldModel.score(true))

// Probability of George winning when considering only Bob's illness
var probGeorgeWinsBobIll = Math.exp(distKerenWorldModel.score(true))


// Explanation score could be the difference in these probabilities
var expl_score = 0.33

console.log(expl_score)

/////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////// bob is ill

var distJaneWorldModel = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   JaneWorldModel)
viz(distJaneWorldModel)

var distAnnaKerenWorldModel = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   AnnaKerenWorldModel)
viz(distAnnaKerenWorldModel)

// Probability of George winning when considering both Bob's illness and Anna's weakness
var probGeorgeWinsAnnaWeak = Math.exp(distJaneWorldModel.score(true))

// Probability of George winning when considering only Bob's illness
var probGeorgeWinsBobIll = Math.exp(distKerenWorldModel.score(true))

// Explanation score could be the difference in these probabilities
var expl_score = 0.3

console.log(expl_score)

/////////////////////////////////////////////////////////////

var softmax = function(scores) {
    var maxScore = Math.max.apply(null, scores)
    var expScores = map(function(score) { return Math.exp(score - maxScore) }, scores) // Subtract max for numerical stability
    var sumExpScores = sum(expScores)
    return map(function(expScore) { return expScore / sumExpScores }, expScores)
}

// Example scores (replace with actual computed scores)
var relevanceScores = [0.33, 0] // Replace with your relevance scores
var utilityScores = [0,0]


var combinedScores = map2(function(u, r) { return u + r; }, relevanceScores, utilityScores)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)


// Jane

var softmax = function(scores) {
    var maxScore = Math.max.apply(null, scores)
    var expScores = map(function(score) { return Math.exp(score - maxScore) }, scores) // Subtract max for numerical stability
    var sumExpScores = sum(expScores)
    return map(function(expScore) { return expScore / sumExpScores }, expScores)
}

// Example scores (replace with actual computed scores)
var relevanceScores = [0, 0.3] // Replace with your relevance scores
var utilityScores = [0,0]


var combinedScores = map2(function(u, r) { return u + r; }, relevanceScores, utilityScores)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)

