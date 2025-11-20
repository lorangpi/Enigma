(define (domain assembly)
(:requirements :strips :typing)
(:types obj)
(:predicates
(p1 ?obj1 ?obj2 - obj) ; (on ?obj2 ?obj1)
)

(:action MOVE ; Move ?obj2 from ?obj1 to ?obj3
:parameters (?obj1 ?obj2 ?obj3 - obj)
:precondition (and 
(p1 ?obj2 ?obj1)
)
:effect (and 
(not (p1 ?obj2 ?obj1)) 
(p1 ?obj2 ?obj3) 
))
)