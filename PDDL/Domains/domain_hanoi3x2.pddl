
(define (domain hanoi)
  (:requirements :strips :typing)

(:predicates
  (p1 ?o - object ?o - object)
  (p0 ?o - object)
  (b0 ?o - object ?o - object)
  (b1 ?o - object ?o - object)

)

  (:action a0
    :parameters (?obj0 - object ?obj1 - object ?obj2 - object)
    :precondition (and (b0 ?obj1 ?obj0) (p0 ?obj0) (p0 ?obj1) (not (p0 ?obj2)) (p1 ?obj1 ?obj2) (not (p1 ?obj1 ?obj0)))
    :effect (and 
      (not (p0 ?obj0)) (p0 ?obj2) (p1 ?obj1 ?obj0) (not (p1 ?obj1 ?obj2))
    )
  )
)

;meta-feature mk0: atom=p0(?obj0), used-by={a0}
;meta-feature mk1: atom=p0(?obj1), used-by={a0}
;meta-feature mk2: atom=p0(?obj2), used-by={a0}
;meta-feature mk3: atom=p1(?obj0,?obj1), used-by={a0}
;meta-feature mk4: atom=p1(?obj0,?obj2), used-by={a0}
;meta-feature mk5: atom=p1(?obj1,?obj2), used-by={a0}

;Action in terms of meta-features:
;action a0: label=MOVE, pre={-mk1,-mk2,mk3,-mk4}, eff={-mk0,mk1,mk4,-mk5}, aff={mk0,mk1,mk4,mk5}

;Action schema:
;action a0 MOVE(?obj0,?obj1,?obj2): static-pre={b0(?obj0,?obj1),b0(?obj0,?obj2),b0(?obj1,?obj2),b1(?obj0,?obj2),b1(?obj1,?obj2)}, pre={-p0(?obj1),-p0(?obj2),p1(?obj0,?obj1),-p1(?obj0,?obj2)}, eff={-p0(?obj0),p0(?obj1),p1(?obj0,?obj2),-p1(?obj1,?obj2)}
