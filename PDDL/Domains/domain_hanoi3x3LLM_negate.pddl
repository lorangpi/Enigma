(define (domain hanoi)
  (:requirements :strips :typing)
  (:types obj)
  
  (:predicates
    (p0 ?obj - obj)
    (p1 ?obj1 ?obj2 - obj)
    (b0 ?obj1 ?obj2 - obj)
    (b1 ?obj1 ?obj2 - obj)
  )

  (:action MOVE
    :parameters (?obj0 ?obj1 ?obj2 - obj)
    :precondition (and 
      ;(not (p0 ?obj0))
      (p0 ?obj1)
      (p0 ?obj2) 
      (not (p1 ?obj0 ?obj1))
      ;(not (p1 ?obj1 ?obj2))
      (p1 ?obj0 ?obj2)
      ;(b0 ?obj0 ?obj1) 
      ;(b0 ?obj0 ?obj2) 
      ;(b0 ?obj1 ?obj2) 
      ;(b1 ?obj0 ?obj2) 
      ;(b1 ?obj1 ?obj2)
    )
    :effect (and 
      (p0 ?obj0)
      (not (p0 ?obj1) )
      (not (p1 ?obj0 ?obj2) )
      (p1 ?obj1 ?obj2)
    )
  )
)