(define (domain hanoi)
(:requirements :strips :typing)
(:types obj)
(:predicates
(p1 ?obj1 ?obj2 - obj) ; (on ?obj2 ?obj1)
(p2 ?obj1 ?obj2 - obj) ; static (not (equal ?obj1 ?obj2))
(p3 ?obj1 ?obj2 - obj) ; static ?obj1 > ?obj2
)

(:action MOVE ; Move ?obj2 from ?obj0 to ?obj1 [ Hence, ?obj2 must be smaller than ?obj1 ]
:parameters (?obj0 ?obj1 ?obj2 - obj)
:precondition (and 
(p1 ?obj0 ?obj0)
(p1 ?obj1 ?obj1)
(p1 ?obj1 ?obj2)
)
:effect (and 
(not (p1 ?obj0 ?obj0)) 
(not (p1 ?obj1 ?obj2)) 
(p1 ?obj1 ?obj0) 
(p1 ?obj2 ?obj2) 
))
)