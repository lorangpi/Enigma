(define (problem hanoi-problem)
(:domain hanoi)
(:objects o1 o2 o3 o4 o5 o6 o7 - obj)
(:init 
(p1 o6 o6 )
(p1 o2 o2 )
(p1 o1 o1 )
(p1 o6 o7 )
(p1 o2 o5 )
(p1 o1 o3 )
(p1 o7 o4 )

(p3 o2 o1)
(p3 o3 o1)
(p3 o4 o1)
(p3 o5 o1)
(p3 o6 o1)
(p3 o7 o1)

(p3 o3 o2)
(p3 o4 o2)
(p3 o5 o2)
(p3 o6 o2)
(p3 o7 o2)

(p3 o1 o3)

(p3 o2 o4)
(p3 o3 o4)

(p3 o3 o6)
(p3 o4 o6)
(p3 o5 o6)
(p3 o7 o6)

(p3 o3 o7)
(p3 o4 o7)
(p3 o5 o7)

)
(:goal (and 

(p1 o1 o1)
(p1 o1 o2)
(p1 o2 o6)
(p1 o6 o7)
(p1 o7 o5)
(p1 o3 o3)
(p1 o4 o4)
))
)