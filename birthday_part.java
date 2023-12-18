//////////////////////////////////////////////////////////////////////////////

var business = mem(function(person) {
  //most people are not busy and can turn up to most parties. Few people are quite busy and a few are very busy.
  var a = sample(Categorical({ps:[0.9,0.05,0.05], vs:[0.1,0.5,0.9]}))
  return a
})

var busy = mem(function(party,person) {
// tells for each party, for each person whether they are busy or not
  return flip(business(person))
})

var turnup = mem(function(party,person) {
  return !busy(party,person)
})

var MyWorldModel_cake = function() {
    condition(business("Lucy")==0.9)  
    condition(turnup("bday","Lucy")==false)
  return turnup("bday","Lucy")
}

var MyWorldModel_future = function() {
    condition(business("Lucy")==0.9)  
    condition(turnup("bday","Lucy")==false)
  return turnup("next_month","Lucy")
}

var PresentAnnabellaWorldModel = function() {
  return turnup("bday","Lucy")
}

var FutureAnnabellaWorldModel = function() {
  return turnup("next_month","Lucy")
}

var concert_Model_for_cake = function() {  
    condition(turnup("bday","Lucy")==false)
  return turnup("bday","Lucy")
}

var busy_Model_for_cake = function() {  
    condition(business("Lucy")==0.9)
  return turnup("bday","Lucy")
}

var busy_Model_for_future = function() {  
    condition(business("Lucy")==0.9)  
  return turnup("next_month","Lucy")
}

var concert_Model_for_future = function() {  
    condition(turnup("bday","Lucy")==false)
  return turnup("next_month","Lucy")
}

///////////////////////////////////////////////////////////////////////

var distCake = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   PresentAnnabellaWorldModel)

var distConcertCake = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   concert_Model_for_cake )
// viz(distConcertFuture)

var distBusyCake = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
    busy_Model_for_cake )
// viz(distFutureBusy)


var concert_cake_expl_util = Math.exp(distConcertCake.score(false))-Math.exp(distCake.score(false))
var busy_cake_expl_util =  Math.exp(distBusyCake.score(false))-Math.exp(distCake.score(false))
var util_present = {"busy":busy_cake_expl_util,"concert":concert_cake_expl_util}

console.log('busy utility present:')
console.log(util_present["busy"])

console.log('concert utility present:')
console.log(util_present["concert"])

///////////////////////////////////////////////////////////////////////

var distFuture = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   FutureAnnabellaWorldModel)
// viz(distFuture)

var distConcertFuture = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
   concert_Model_for_future )
// viz(distConcertFuture)

var distFutureBusy = Infer({method: 'MCMC', kernel: 'MH', samples: 60000},
    busy_Model_for_future )
// viz(distFutureBusy)


// for utility for future scenario is that an explanation should give information that is useful in the future.
// therefore, utility now is proportional to explanation power of a reason for predicting whether Lucy will turn up to the party next month
var concert_expl_util = Math.exp(distConcertFuture.score(false))-Math.exp(distFuture.score(false))
var busy_expl_util =  Math.exp(distFutureBusy.score(false))-Math.exp(distFuture.score(false))
var util_next_month = {"busy":busy_expl_util,"concert":concert_expl_util}

console.log('busy utility:')
console.log(util_next_month["busy"])

console.log('concert utility:')
console.log(util_next_month["concert"])

/////////////////////////////////////////////////////////////

var softmax = function(scores) {
    var maxScore = Math.max.apply(null, scores)
    var expScores = map(function(score) { return Math.exp(score - maxScore) }, scores) // Subtract max for numerical stability
    var sumExpScores = sum(expScores)
    return map(function(expScore) { return expScore / sumExpScores }, expScores)
}

// Example scores (replace with actual computed scores)
var utilityScores = [util_next_month["busy"], util_next_month["concert"]] // Replace with your relevance scores
var relevance = [util_present["busy"], util_present["concert"]] // Replace with your utility scores

console.log('utilities:', utilityScores)

var combinedScores = map2(function(r, u) { return u + r }, relevance, utilityScores)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)

// cake

var utilityScores = [0, 0] // Replace with your relevance scores
var relevance = [util_present["busy"], util_present["concert"]] // Replace with your utility scores

var combinedScores = map2(function(r, u) { return u + r }, relevance, utilityScores)

var softmaxProbabilities = softmax(combinedScores)

console.log(softmaxProbabilities)

