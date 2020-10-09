using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using UnityEngine;


namespace Unity.MLAgents.Extensions.Match3
{
    public class Match3Actuator : IActuator
    {
        private AbstractBoard m_Board;
        private ActionSpec m_ActionSpec;
        private bool m_ForceRandom;
        private System.Random m_Random;

        private int m_Rows;
        private int m_Columns;
        private int m_NumCellTypes;

        public Match3Actuator(AbstractBoard board, bool forceRandom, int randomSeed)
        {
            m_Board = board;
            m_Rows = board.Rows;
            m_Columns = board.Columns;
            m_NumCellTypes = board.NumCellTypes;

            m_ForceRandom = forceRandom;
            if (forceRandom)
            {
                m_Random = new System.Random(randomSeed);
            }
            var numMoves = Move.NumPotentialMoves(m_Board.Rows, m_Board.Columns);
            m_ActionSpec = ActionSpec.MakeDiscrete(numMoves);
        }

        public ActionSpec ActionSpec => m_ActionSpec;

        public void OnActionReceived(ActionBuffers actions)
        {
            int moveIndex = 0;
            if (m_ForceRandom)
            {
                moveIndex = m_Board.GetRandomValidMoveIndex(m_Random);
            }
            else
            {
                moveIndex = actions.DiscreteActions[0];
            }

            if (m_Board.Rows != m_Rows || m_Board.Columns != m_Columns || m_Board.NumCellTypes != m_NumCellTypes)
            {
                Debug.LogWarning(
                    $"Board shape changes since actuator initialization. This may cause unexpected results. " +
                    $"Old shape: Rows={m_Rows} Columns={m_Columns}, NumCellTypes={m_NumCellTypes} " +
                    $"Current shape: Rows={m_Board.Rows} Columns={m_Board.Columns}, NumCellTypes={m_Board.NumCellTypes}"
                );
            }

            Move move = Move.FromMoveIndex(moveIndex, m_Rows, m_Columns);
            m_Board.MakeMove(move);
        }

        public void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
        {
            using (TimerStack.Instance.Scoped("WriteDiscreteActionMask"))
            {
                actionMask.WriteMask(0, InvalidMoveIndices());
            }
        }

        public string Name => "Match3Actuator";// TODO pass optional name

        public void ResetData()
        {
        }

        IEnumerable<int> InvalidMoveIndices()
        {
            foreach (var move in m_Board.InvalidMoves())
            {
                yield return move.MoveIndex;
            }
        }
    }
}
