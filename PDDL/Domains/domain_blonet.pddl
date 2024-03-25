(define (domain hanoi)
(:requirements :strips :typing)
(:types obj)
(:predicates
(p0 ?obj - obj) ; (not (clear ?obj))
(p1 ?obj1 ?obj2 - obj) ; (not (on ?obj2 ?obj1))
(b0 ?obj1 ?obj2 - obj) ; static (not (equal ?obj1 ?obj2))
(b1 ?obj1 ?obj2 - obj) ; static ?obj1 > ?obj2
)

(:action MOVE ; Move ?obj2 from ?obj0 to ?obj1 [ Hence, ?obj2 must be smaller than ?obj1 ]
:parameters (?obj0 ?obj1 ?obj2 - obj)
:precondition (and 
(not (p0 ?obj1)) ; (clear ?obj1)
(not (p0 ?obj2)) ; (clear ?obj2)
(p1 ?obj0 ?obj1) ; (not (on ?obj1 ?obj0))
(not (p1 ?obj0 ?obj2)) ; (on ?obj2 ?obj0)
; Here I ignore static predicates to relax the action conditions, for testing purposes
;(b0 ?obj0 ?obj1) ; ?obj0 != ?obj1
;(b0 ?obj0 ?obj2) ; ?obj0 != ?obj2
;(b0 ?obj1 ?obj2) ; ?obj1 != ?obj2
;(b1 ?obj0 ?obj2) ; ?obj0 > ?obj2 [ REDUNDANT ]
;(b1 ?obj1 ?obj2) ; ?obj1 > ?obj2
)
:effect (and 
(not (p0 ?obj0)) ; (clear ?obj0)
(p0 ?obj1) ; (not (clear ?obj1))
(p1 ?obj0 ?obj2) ; (not (on ?obj2 ?obj0))
(not (p1 ?obj1 ?obj2)) ; (on ?obj2 ?obj1)
))
)