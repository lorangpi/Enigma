(define (domain kitchen)
(:requirements :strips :typing)
(:types obj)
(:predicates
(p1 ?obj1 ?obj2 - obj) ; (on ?obj2 ?obj1)
(p2); (stove on)
(p3 ?obj1 ?obj2 - obj); static ?obj1 > ?obj2
(p4); cooked
)

(:action MOVE ; Move ?obj2 from ?obj1 to ?obj3
:parameters (?obj1 ?obj2 ?obj3 - obj)
:precondition (and 
(p1 ?obj2 ?obj1)
(p3 ?obj2 ?obj1)
)
:effect (and 
(not (p1 ?obj2 ?obj1)) 
(p1 ?obj2 ?obj3) 
))

(:action TURNON 
:parameters ()
:precondition (and 
(not (p2)) 
)
:effect (and 
(p2) 
))

(:action TURNOFF 
:parameters ()
:precondition (and 
(p2)
)
:effect (and 
(not (p2)) 
))

(:action WAIT ; cook
:parameters (?obj1)
:precondition (and 
(p2)
(p1 o2 o3)
(p1 ?obj1 o2)
)
:effect (and 
(p4) 
))


)