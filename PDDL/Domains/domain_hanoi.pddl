(define (domain hanoi)
  (:requirements :strips :typing)
(:types 
  object
  disk peg - object
)
  
(:predicates
  (on ?d - disk ?o - object)
  (clear ?o - object)
  (smaller ?d - disk ?o - object)
)

  (:action move
    :parameters (?d - disk ?from - object ?to - object)
    :precondition (and (on ?d ?from) (clear ?d) (clear ?to) (smaller ?d ?to))
    :effect (and 
      (not (on ?d ?from)) 
      (not (clear ?to)) 
      (on ?d ?to) 
      (clear ?from)
    )
  )
)