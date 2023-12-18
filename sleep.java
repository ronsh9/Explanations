//reason for sleeping: 1. rowing practice and 2. boring

var anonymous_util = {"rowing":0,"boring":0.5}  //Mr Tod wants to improve
var direct_util = {"rowing":0,"boring":-1}  //It will be awkward

//////////////////////////////////////////////////////////

var sleep = function(boring,rowing) {
  // proability of going to sleep as a function of boringness and whether I went rowing
  if (boring & rowing){
    return true
  }
  if (rowing){
    return flip(0.5)
  }
  if (boring){
    return flip(0.8)
  }
  return false
}

/////////////////////////////////////////////////////////////

var MyWorldModel = function() {
  var boring = true
  var rowing = true
  return sleep(boring,rowing)
}

var ToddModel = function() {
  var boring = flip(0.1)// he thinks he is not boring
  var rowing = flip(0.1)// probabity a given person is a rower (or some other sport?) is 0.1
  return sleep(boring,rowing)
}

var boringToddModel = function() {
  var boring = true //  you tell him he is boring
  var rowing = flip(0.1)// probabity a given person is a rower (or some other sport?) is 0.1
  return sleep(boring,rowing)
}

var rowingToddModel = function() {
  var boring = flip(0.1)// he thinks he is not boring
  var rowing = true// "I have rowing practice"
  return sleep(boring,rowing)
}


//////////////////////////////////////////////////////////////


var distOrigTodd = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   ToddModel)
var distBoringTodd = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   boringToddModel)
var explanation_score_boring = Math.exp(distBoringTodd.score(true))- Math.exp(distOrigTodd.score(true))
console.log('boring pragmatic explanation score:')
console.log(explanation_score_boring)


var distOrigTodd = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   ToddModel)
var distRowingTodd = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   rowingToddModel)
var explanation_score_rowing = Math.exp(distRowingTodd.score(true))- Math.exp(distOrigTodd.score(true))
console.log('rowing pragmatic explanation score:')
console.log(explanation_score_rowing)

var relevance_score = {"boring": explanation_score_boring,"rowing": explanation_score_rowing}


/////////////////////////////////////////////////////////////

var softmax = function(scores) {
    var maxScore = Math.max.apply(null, scores)
    var expScores = map(function(score) { return Math.exp(score - maxScore) }, scores) // Subtract max for numerical stability
    var sumExpScores = sum(expScores)
    return map(function(expScore) { return expScore / sumExpScores }, expScores)
}

// Example scores (replace with actual computed scores)
var relevance_scores = [relevance_score["boring"], relevance_score["rowing"]] // Replace with your relevance scores
var util = [anonymous_util["boring"], anonymous_util["rowing"]] // Replace with your utility scores

var combinedScores = map2(function(r, u) { return 5*u + r; }, relevance_scores, util)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)

// direct

var relevance_scores = [relevance_score["boring"], relevance_score["rowing"]] // Replace with your relevance scores
var util = [direct_util["boring"], direct_util["rowing"]] // Replace with your utility scores

var combinedScores = map2(function(r, u) { return 2.5*u + r; }, relevance_scores, util)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)

