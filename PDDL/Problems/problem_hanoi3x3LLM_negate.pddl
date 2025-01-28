(define (problem hanoi-problem)
(:domain hanoi)
(:objects o0 o1 o2 o3 o4 o5 - obj)
  (:init
    (p0 o3)
    (p0 o4)
    (p0 o5)
    (p1 o2 o3)
    (p1 o1 o2)
    (p1 o0 o1)
    )

  (:goal (and
    (p0 o3)
    (p0 o4)
    (p0 o0)

    (p1 o2 o3)
    (p1 o1 o2)
    (p1 o5 o1)
  ))
  )